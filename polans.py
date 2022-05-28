from datetime import date
from re import L
from obspy import read
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
from obspy.signal.polarization import polarization_analysis
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stream
from obspy.signal.filter import bandpass
from matplotlib.dates import date2num
import matplotlib.pyplot as plt 
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
from numpy import cos, sin, angle, where, array, median, abs, fft, argsort, rint, sqrt, real;
import click
from os.path import basename,dirname
from os import sep
import os
import numpy as np
import obspy

def spectrum(trace):
    time_step = 1.0/trace.stats.sampling_rate
    ps = abs(fft.fft(trace.data))**2
    ps = sqrt(ps)
    freqs = fft.fftfreq(trace.data.size, time_step)
    idx = argsort(freqs)
    fresult = freqs[idx]
    psresult = ps[idx]
    l = int(len(fresult)/2)
    return fresult[l:], psresult[l:]

def makeWindow(tr,winlen=30):
    rectime = tr.stats.endtime-tr.stats.starttime
    if (rectime < 1200):
        print("Record Time {} minutes, need more than 20 minutes".format(rectime/60))
        return None
    print("Creating {} second windows on {} minutes record time".format(winlen,rectime/60))
    print("Skipping first 10 minutes")
    startwindow = tr.stats.starttime+600
    endwindow = startwindow + winlen
    windowlist = []
    print("Skipping last 10 minutes")
    while(endwindow < tr.stats.endtime-600):
        windowlist.append((startwindow, endwindow))
        startwindow += winlen
        endwindow += winlen
    print("{} windows created, {} minutes in total".format(len(windowlist),len(windowlist)*winlen/60))
    return windowlist

def trigWindow(tr, winlen=20, sta=1, lta=30, head=1.25, tail=1.0):
    wd = makeWindow(tr, winlen)
    df = tr.stats.sampling_rate 
    print("Detect transient noise data on every window")
    # cft = classic_sta_lta(tr.data/np.std(np.abs(tr.data)), int(sta * df), int(lta * df))
    _tmp_data = tr.data/np.std(np.abs(tr.data))
     # STA/LTA calculation on a baseline corrected data
    cft = recursive_sta_lta((_tmp_data-np.mean(_tmp_data)), int(sta * df), int(lta * df))
    on_of = trigger_onset(cft, head, tail)
    badwindow = []
    lastWindowIndex = -1
    for item in on_of:
        startT = tr.stats.starttime + item[0]/tr.stats.sampling_rate
        endT = tr.stats.starttime + item[1]/tr.stats.sampling_rate
        triggered = False
        for i, w in enumerate(wd):
            if(i>lastWindowIndex):
                if(startT>w[0] and startT<w[1] and not triggered):
                    triggered = True
                    badwindow.append(i)
                    lastWindowIndex = i
                elif(endT>w[1] and triggered):
                    badwindow.append(i)
                    lastWindowIndex = i
                elif(endT<w[1] and triggered):
                    badwindow.append(i)
                    lastWindowIndex = i
                    triggered = False
                    break
    badwindow.reverse()
    print("{} windows with transient noise detected, removing windows".format(len(badwindow)))
    for x in badwindow:
        wd.pop(x)
    print("windows removed, data reduced to {:.0f} minutes".format(winlen*len(wd)/60))
    return wd

def streamCheck(st):
    if st.select(component="Z") and st.select(component="N") and st.select(component="E"):
        return True
    else:
        return False

def rotate90(inp):
    return abs(90 - inp)

def polarization(st,winlen=30):
    if not streamCheck:
        return None
    print("Begin Polarization Analysis")
    result = polarization_analysis(st,winlen,1,0,10,st[0].stats.starttime+600,st[0].stats.endtime-600,False,'flinn')
    for i,item in enumerate(result['incidence']):
       result['incidence'][i] = rotate90(item) 
    # print([UTCDateTime(t) for t in result['timestamp']])
    return result

def azimuthRotate(inp, rotation):
    lst = []
    for i in range(len(inp)):
        val = inp[i]-rotation
        val = 180+val if val<0 else val
        lst.append(val)
    return array(lst)

def azimuthStd(inp, wind):
    lst = []
    inp90 = azimuthRotate(inp, 90)
    for i in range(len(inp)-wind+1):
        wowL = []
        wowL90 = []
        for j in range(wind):
            wowL.append(inp[i+j])
            wowL90.append(inp90[i+j])
        wow = array(wowL)
        wow90 = array(wowL90)
        lst.append(min(wow.std(),wow90.std()))
        # lst.append(wow90.std())
    for i in range(wind-1):
        lst.append(lst[-1])
    return array(lst)

def azimuthMedis(inp, wind):
    lst = []
    inp90 = azimuthRotate(inp, 90)
    for i in range(len(inp)-wind+1):
        wowL = []
        wowL90 = []
        for j in range(wind):
            wowL.append(inp[i+j])
            wowL90.append(inp90[i+j])
        wow = array(wowL)
        wow90 = array(wowL90)
        # lst.append(min(wow.std(),wow90.std()))
        lst.append(abs(median(wow90)-inp90[i]))
    for i in range(wind-1):
        lst.append(lst[-1])
    return array(lst)

def surfaceNoise(inci, azim, incth=25, azistdth=15):
    lst = []
    for i in range(len(inci)):
        item = 'r' if inci[i]<incth and azim[i]<azistdth else 'k' #note IRNAKA: tadinya 25 diganti 15 agar konsisten dengan azicol = where(azstd<15,'r','k')
        lst.append(item)
    return lst

def calibrate(inpstream,z=1,n=1,e=1):
    if(z==1 and n==1 and e==1):
        return inpstream
    result = Stream()
    tz = inpstream.select(component="Z")[0]
    tn = inpstream.select(component="N")[0]
    te = inpstream.select(component="E")[0]
    tz.data *= z
    tn.data *= n
    te.data *= e
    result.append(tz)
    result.append(tn)
    result.append(te)
    return result

def P2R(A, phi):
    return A * ( cos(phi) + sin(phi)*1j )

def calibrateArray(data, calibration_factor):
    dataf = fft.fft(data)
    _A = abs(dataf) * calibration_factor
    _phi = angle(dataf)
    dataf = P2R(_A,_phi)
    data = fft.ifft(dataf)
    return real(data)

def calibrateF(inpstream,z=1,n=1,e=1):
    if(z==1 and n==1 and e==1):
        return inpstream
    result = Stream()
    tz = inpstream.select(component="Z")[0]
    tn = inpstream.select(component="N")[0]
    te = inpstream.select(component="E")[0]
    tz.data = calibrateArray(tz.data,z)
    tn.data = calibrateArray(tn.data,n)
    te.data = calibrateArray(te.data,e)
    result.append(tz)
    result.append(tn)
    result.append(te)
    return result
    

def plot(test, filename,z=1,n=1,e=1,winlen=10, isexport=False, incth=25, azistdth=15, trigger_on=1.25, trigger_off=1.00):
    test = calibrateF(test,z,n,e)
    test_filtered = test.copy()
    test_filtered.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    if isexport:
        test.write(dirname(filename[0])+filename[0].replace('.mseed','precalibrated_data.mseed'),format="MSEED")
    filename = basename(filename[0])
    tz = test.select(component="Z")[0]
    tn = test.select(component="N")[0]
    te = test.select(component="E")[0]
    windows = trigWindow(tz,winlen,head=trigger_on,tail=trigger_off)
    paz = polarization(test_filtered,winlen)
    azstd = azimuthStd(paz["azimuth"],10)
    incicol = where(paz['incidence']<incth,'r','k')
    azicol = where(azstd<azistdth,'r','k')
    sNoise = surfaceNoise(paz['incidence'],azstd, incth, azistdth)

    # fig, ax = plt.subplots(7,sharex=True)
    fig = plt.figure()
    ax = []
    spec = gridspec.GridSpec(ncols=1,nrows=10,figure=fig,height_ratios=[1,1,1,1, 1,1,0.6,1,0.7,1])
    for i in range(8):
        if i==0:
            axtemp = fig.add_subplot(spec[i, 0])
        elif i==6:
            axtemp = fig.add_subplot(spec[i+1, 0])
        elif i==7:
            axtemp = fig.add_subplot(spec[i+2, 0])
        else:
            axtemp = fig.add_subplot(spec[i, 0],sharex=ax[0])
        ax.append(axtemp)

    fig.set_size_inches(8, 10)
    # fig.suptitle('{}'.format(str(tz).replace('|','\n')), x=0.1,ha="left", fontsize=12)
    efectiveStart = tz.stats.starttime + 600
    efectiveEnd = tz.stats.endtime - 600
    offset = np.max([tz.data.std(),tn.data.std(),te.data.std()])
    ax[0].plot(te.times("matplotlib"),te.data, color='blue',linewidth=0.25, alpha=0.8, label="E")
    ax[0].plot(tn.times("matplotlib"),tn.data+5*offset, color='green',linewidth=0.25, alpha=0.8, label="N")
    ax[0].plot(tz.times("matplotlib"),tz.data+10*offset, color='red',linewidth=0.25, alpha=0.8, label="Z")
    leg0 = ax[0].legend()

    tz.trim(efectiveStart,efectiveEnd)
    tn.trim(efectiveStart,efectiveEnd)
    te.trim(efectiveStart,efectiveEnd)

    zfreq, zspec = spectrum(tz)
    nfreq, nspec = spectrum(tn)
    efreq, espec = spectrum(te)
    gap = max(zfreq)+8
    ax[7].plot(zfreq, zspec, color='red', label="Z")
    ax[7].plot([i+gap for i in nfreq], nspec, color='green', label="N")
    ax[7].plot([i+gap*2 for i in efreq], espec, color='blue', label="E")
    ax[7].set_yscale('log')
    # ax[7].set_xscale('log')
    ax[7].set_xticks([0,gap-8, gap, gap*2-8, gap*2, gap*3-8])
    ax[7].set_xticklabels([0,rint(max(zfreq)),0,rint(max(nfreq)),0,rint(max(efreq))])

    ax[0].set_ylim(min(te.data),max(tz.data)+10*offset)
    offset = np.max([tz.data.std(),tn.data.std(),te.data.std()])
    
    tz.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    tn.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    te.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    ax[1].plot(te.times("matplotlib"),te.data*0.5, color='blue',linewidth=0.25, alpha=0.8, label="E")
    ax[1].plot(tn.times("matplotlib"),tn.data*0.5+5*offset, color='green',linewidth=0.25, alpha=0.8, label="N")
    ax[1].plot(tz.times("matplotlib"),tz.data*0.5+10*offset, color='red',linewidth=0.25, alpha=0.8, label="Z")
    leg1 = ax[1].legend()
    leg7 = ax[7].legend()
    
    for legobj in leg0.legendHandles:
        legobj.set_linewidth(5.0)
    for legobj in leg1.legendHandles:
        legobj.set_linewidth(5.0)
    for legobj in leg7.legendHandles:
        legobj.set_linewidth(5.0)
    # ax[0].legend(linewidth=6)
    # ax[1].legend(linewidth=6)

    
    windowNoSurface = 0
    for i, w in enumerate(windows):
        if len(sNoise)<(i+1):
            break
        windowNoSurface += 1 if sNoise[i] == 'k' else 0
        ax[1].axvspan(w[0].matplotlib_date, w[1].matplotlib_date, alpha=0.5, facecolor=sNoise[i], edgecolor=None)
    print("{} windows with surface noise detected, removing windows".format(len(windows)-windowNoSurface))

    ax[2].axhspan(0,25,alpha=0.3, facecolor='k',edgecolor=None)
    ax[2].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['incidence'], color=incicol,s=1)
    ax[3].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['azimuth'], color=azicol,s=1)
    ax[4].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['rectilinearity'], color='k',s=1)
    ax[5].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['planarity'], color='k',s=1)

    ax[2].set_ylim(-10,100)
    ax[3].set_ylim(-20,200)
    ax[4].set_ylim(-0.1,1.1)
    ax[5].set_ylim(-0.1,1.1)
    ax[6].set_ylim(-4,1)

    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks((0,45,90))
    ax[3].set_yticks((0,90,180))
    ax[4].set_yticks((0,0.5,1))
    ax[5].set_yticks((0,0.5,1))

    ax[0].set_ylabel('Raw Data')
    ax[1].set_ylabel('Filtered(1-5)')
    ax[2].set_ylabel('Incidence')
    ax[3].set_ylabel('Azimuth')
    ax[4].set_ylabel('Rectilinearity')
    ax[5].set_ylabel('Planarity')
    ax[6].set_ylabel('Noise Composition')
    ax[6].set_xlabel('minutes')
    ax[7].set_ylabel('Spectrum')
    ax[7].set_xlabel('Hz')

    # ax[2].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()
    ax[1].yaxis.tick_right()
    ax[2].yaxis.tick_right()
    ax[3].yaxis.tick_right()
    ax[4].yaxis.tick_right()
    ax[5].yaxis.tick_right()
    ax[6].yaxis.tick_right()
    ax[7].yaxis.tick_right()

    # ax[0].xaxis.tick_top()
    # ax[1].xaxis.tick_top()
    # ax[2].xaxis.tick_top()
    # ax[3].xaxis.tick_top()
    # ax[4].xaxis.tick_top()
    # ax[5].xaxis.tick_top()

    # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    # ax[0].xaxis.set_major_formatter(xfmt)

    # ax[0].xaxis.set_major_formatter(
    #     md.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
    # ax[0].xaxis.set_major_locator(md.MinuteLocator())
    # ax[0].xaxis.set_major_formatter(md.DateFormatter('%b %d %H:%M'))
    # ax[0].xaxis.set_minor_formatter(md.DateFormatter('%y-%b-%d'))
    locator = md.AutoDateLocator(minticks=5, maxticks=14)
    formatter = md.ConciseDateFormatter(locator)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(formatter)

    report = "Data Quality Report for " + filename + " Record\n"
    recordingLength = (tz.stats.endtime - tz.stats.starttime+1200)/60
    report += "Record duration : {:.0f} minutes ({:.0%})\n".format(recordingLength, recordingLength/recordingLength)
    report += "Effective Data : {:.0f} minutes ({:.0%})\n".format(recordingLength-20, (recordingLength-20)/recordingLength)
    transientLength = winlen*len(windows)/60
    report += "Transient Free Data : {:.0f} minutes ({:.0%})\n".format(transientLength, transientLength/recordingLength)
    surfaceLength = winlen*windowNoSurface/60
    report += "Surface Noise Free Data : {:.0f} minutes ({:.0%})\n".format(surfaceLength, surfaceLength/recordingLength)
    dataQ = "POOR"
    if surfaceLength>85:
        dataQ = "GOOD"
        tobeGood = surfaceLength-85
    elif surfaceLength>50:
        dataQ = "FAIR"
        tobeFair = surfaceLength-85
    else:
        dataQ = "POOR"
    report += "Data Quality : " + dataQ + " -> {:.0f} minutes".format(surfaceLength)
    if surfaceLength<50:
        report += ", ({:.0f}/{:.0f}) more minutes to be (fair/good)".format(abs(surfaceLength-50),abs(surfaceLength-85))
    elif surfaceLength<85:
        report += ", {:.0f} more minutes to be good".format(abs(surfaceLength-85))

    fig.suptitle(report, x=0.1,ha="left", fontsize=10)
    plt.subplots_adjust(hspace=.0)

    noiseSource = ['set up','transient', 'incidence', 'azimuth']
    ypos = [[-i,-i] for i in range(len(noiseSource))]
    startBar = 0
    setupNoise = 1200/60
    transientNoise = (recordingLength-20-transientLength)
    incidenceNoise = 0
    azimuthNoise = 0
    for i in range(len(incicol)):
        incidenceNoise += 1 if incicol[i]=='r' else 0
        azimuthNoise += 1 if azicol[i]=='r' else 0
    incidenceNoise *= winlen
    azimuthNoise *= winlen
    incidenceNoise = incidenceNoise/60
    azimuthNoise = azimuthNoise/60
    ax[6].set_yticks([0,-1,-2,-3])
    ax[6].set_yticklabels(noiseSource)
    ax[6].plot((startBar,setupNoise),ypos[0],linewidth=5)
    ax[6].plot((startBar,transientNoise),ypos[1],linewidth=5)
    ax[6].plot((startBar,incidenceNoise),ypos[2],linewidth=5)
    ax[6].plot((startBar,azimuthNoise),ypos[3],linewidth=5)

    print("Generate Data Quality Control Report to "+filename+".png")
    plt.savefig(filename+'.png')
    # plt.show()

def combine(filelist):
    st = Stream()
    for item in filelist:
        st += read(item)
    latestStart = UTCDateTime(0)
    earliestend = UTCDateTime(2147483647)
    st.merge(fill_value='latest')
    for tr in st:
        latestStart = tr.stats.starttime if latestStart < tr.stats.starttime else latestStart
        earliestend = tr.stats.endtime if earliestend > tr.stats.endtime else earliestend
    st.trim(latestStart, earliestend)

    return st

def parseCalOption(calstring):
    z = int(calstring[0])
    n = int(calstring[1])
    e = int(calstring[2])
    return [z,n,e]

def calibrateC(filename,calibrator="",target_frequency=[1,5],min_number_of_cycle=100,statistic_mode="mean"):
    calibratorid = np.nan
    data = []
    _min_time = []
    _max_time = []
    for i,instrument in enumerate(filename):
        if 'merged' in instrument:
            if calibrator in instrument and not(calibrator==""): calibratorid=i
            data.append(obspy.read(instrument))
            for j,component in enumerate(data[-1]):
                _min_time.append(component.stats.starttime)
                _max_time.append(component.stats.endtime)

    if len(_min_time)<1:
        raise IOError("error! ensure all files contain 'merged'!")

    starttime = max(_min_time)
    endtime = min(_max_time)

    if starttime>endtime: raise IOError("error! end time is earlier than the start time. Input files are not consistent!")

    print("===================================")
    if statistic_mode=="mean":
        print("Calculating mean calibration factor ± standard deviation")
    elif statistic_mode=="median":
        print("Calculating median calibration factor ± semi-IQR")
    print(f"START TIME: {starttime}")
    print(f"END   TIME: {endtime}")

    for i,instrument in enumerate(data):
        data[i].trim(starttime,endtime)

    time_window = (1/min(target_frequency))*int(min_number_of_cycle)
    total_recording_overlap = endtime-starttime
    number_of_window = int(np.floor(total_recording_overlap/time_window))
    number_of_instrument = len(data)

    cal = np.empty((number_of_instrument,3,number_of_window))+np.nan
    cal_std = np.empty((number_of_instrument,3,number_of_window))+np.nan
    _calibration_factor = np.empty((number_of_instrument,3))+np.nan
    _calibration_factor_std = np.empty((number_of_instrument,3))+np.nan
    _calibration_factor_q1 = np.empty((number_of_instrument,3))+np.nan
    _calibration_factor_q3 = np.empty((number_of_instrument,3))+np.nan
    calibration_factor = np.empty((number_of_instrument,3))+np.nan
    calibration_factor_std = np.empty((number_of_instrument,3))+np.nan
    calibration_factor_q1 = np.empty((number_of_instrument,3))+np.nan
    calibration_factor_q3 = np.empty((number_of_instrument,3))+np.nan
    fftfreq = np.fft.fftfreq(int(time_window*data[0][0].stats.sampling_rate)+1,d=data[0][0].stats.delta)
    startwin = np.max(np.where(fftfreq[:int(len(fftfreq)/2)]<1))
    endwin = np.min(np.where(fftfreq[:int(len(fftfreq)/2)]>5))

    list_component = ['Z','N','E']

    for ilc,lc in enumerate(list_component): 
        # calibrating Z component
        for iw in range(number_of_window):
            _starttime = starttime+(iw*time_window)
            _endtime = starttime+((iw+1)*time_window)
            _freq_component = []
            _calfreq_component = []
            for ii,instrument in enumerate(data):
                trimmed_instrument = instrument.copy()
                trimmed_instrument.trim(_starttime,_endtime)
                for ic,component in enumerate(trimmed_instrument):
                    if lc in component.stats.component:
                        _freq_component.append(np.abs(np.fft.fft(component.data)[startwin:endwin]))
            _mean_amplitude = np.mean(_freq_component,axis=0)
            for ifreq,freqc in enumerate(_freq_component):
                _calfreq_component.append(_mean_amplitude/freqc)

            for ii,instrument in enumerate(data):
                cal[ii,ilc,iw] = np.median(_calfreq_component[ii])
                cal_std[ii,ilc,iw] = np.std(_calfreq_component[ii])*100/cal[ii,ilc,iw]

        for ii in range(number_of_instrument):
            if statistic_mode=="mean":
                _calibration_factor[ii,ilc] = np.mean(cal[ii][ilc])
                _calibration_factor_std[ii,ilc] = np.std(cal[ii][ilc])
                if np.isnan(calibratorid):
                    calibration_factor[ii,ilc] = np.mean(cal[ii][ilc])
                    calibration_factor_std[ii,ilc] = np.std(cal[ii][ilc])
                else:
                    calibration_factor[ii,ilc] = np.mean(cal[ii][ilc]/cal[calibratorid][ilc])
                    calibration_factor_std[ii,ilc] = np.std(cal[ii][ilc]/cal[calibratorid][ilc])
            elif statistic_mode=="median":
                _calibration_factor[ii,ilc] = np.median(cal[ii][ilc])
                _calibration_factor_q1[ii,ilc] = np.percentile(cal[ii][ilc],25)
                _calibration_factor_q3[ii,ilc] = np.percentile(cal[ii][ilc],75)
                if np.isnan(calibratorid):
                    calibration_factor[ii,ilc] = np.median(cal[ii][ilc])
                    calibration_factor_q1[ii,ilc] = np.percentile(cal[ii][ilc],25)
                    calibration_factor_q3[ii,ilc] = np.percentile(cal[ii][ilc],75)
                else:
                    calibration_factor[ii,ilc] = np.median(cal[ii][ilc]/cal[calibratorid][ilc])
                    calibration_factor_q1[ii,ilc] = np.percentile(cal[ii][ilc]/cal[calibratorid][ilc],25)
                    calibration_factor_q3[ii,ilc] = np.percentile(cal[ii][ilc]/cal[calibratorid][ilc],75)

    for ii in range(number_of_instrument):
        if statistic_mode=="mean":
            print(f"Alat{ii+1:02d} Z:{calibration_factor[ii,0]:7.4f} ±{calibration_factor_std[ii,0]:7.4f}   N:{calibration_factor[ii,1]:7.4f} ±{calibration_factor_std[ii,1]:7.4f}   E:{calibration_factor[ii,2]:7.4f} ±{calibration_factor_std[ii,2]:7.4f} {filename[ii]}")
        elif statistic_mode=="median":
            # calculating semi interquartile range
            _siqrz = (calibration_factor_q3[ii,0]-calibration_factor_q1[ii,0])/2
            _siqrn = (calibration_factor_q3[ii,1]-calibration_factor_q1[ii,1])/2
            _siqre = (calibration_factor_q3[ii,2]-calibration_factor_q1[ii,2])/2
            print(f"Alat{ii+1:02d} Z:{calibration_factor[ii,0]:7.4f} ±{_siqrz:7.4f}   N:{calibration_factor[ii,1]:7.4f} ±{_siqrn:7.4f}   E:{calibration_factor[ii,2]:7.4f} ±{_siqre:7.4f} {filename[ii]}")
            # print(f"Alat{ii+1:02d} Z:{calibration_factor[ii,0]:7.4f} Q1:{calibration_factor_q1[ii,0]:7.4f} Q3:{calibration_factor_q3[ii,0]:7.4f}   N:{calibration_factor[ii,1]:7.4f} Q1:{calibration_factor_q1[ii,1]:7.4f} Q3:{calibration_factor_q3[ii,1]:7.4f}   E:{calibration_factor[ii,2]:7.4f} Q1:{calibration_factor_q1[ii,2]:7.4f} Q3:{calibration_factor_q3[ii,2]:7.4f}")
    print("===================================")

@click.command()
@click.argument('filename', type=click.Path(exists=True),nargs=-1)
@click.option('--mode','-m',type=str,default="QC")
@click.option('--calibration','-c')
@click.option('--zfactor', '-z', type=float)
@click.option('--nfactor', '-n', type=float)
@click.option('--efactor', '-e', type=float)
@click.option('--export/--no-export', default=False)
@click.option('--calibrator','-k',type=str,default="")
@click.option('--trigger_on',type=float,default=2.5)
@click.option('--trigger_off',type=float,default=1.2)
def main(filename,mode,calibration,zfactor,nfactor,efactor,export,calibrator,trigger_on,trigger_off):
    if mode=="QC":
        st = combine(filename)
        isexport = False
        if export: isexport=True
        if not calibration and not zfactor and not nfactor and not efactor:
            plot(st, filename,  trigger_on=trigger_on, trigger_off=trigger_off)
        else:
            if calibration:
                [z,n,e] = parseCalOption(calibration)
            else:
                z=1
                n=1
                e=1
            z = zfactor if zfactor else z
            n = nfactor if nfactor else n
            e = efactor if efactor else e
            plot(st, filename, z, n, e, incth=25, azistdth=15, isexport=isexport, trigger_on=trigger_on, trigger_off=trigger_off)
    elif mode=="CALIBRATION":
        calibrateC(filename,calibrator=calibrator,target_frequency=[1,5],min_number_of_cycle=100,statistic_mode="mean")
        print()
        calibrateC(filename,calibrator=calibrator,target_frequency=[1,5],min_number_of_cycle=100,statistic_mode="median")
    else:
        print("Not yet implemented!")
if __name__ == '__main__':
    main()