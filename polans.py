from datetime import date
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.signal.polarization import polarization_analysis
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stream
from matplotlib.dates import date2num
import matplotlib.pyplot as plt 
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
from numpy import cos, sin, angle, where, array, median, abs, fft, argsort, rint, sqrt, real;
import click
from os.path import basename,dirname
from os import sep

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

def trigWindow(tr, winlen=30, sta=5, lta=10, head=1.5, tail=0.7):
    wd = makeWindow(tr, winlen)
    df = tr.stats.sampling_rate 
    print("Detect transient noise data on every window")
    cft = classic_sta_lta(tr.data, int(sta * df), int(lta * df))
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

def surfaceNoise(inci, azim):
    lst = []
    for i in range(len(inci)):
        item = 'r' if inci[i]<25 and azim[i]<25 else 'k'
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
    

def plot(test, filename,z=1,n=1,e=1,winlen=20, isexport=False):
    test = calibrateF(test,z,n,e)
    if isexport:
        test.write(dirname(filename[0])+sep+'precalibrated_data.mseed',format="MSEED")
    filename = basename(filename[0])
    tz = test.select(component="Z")[0]
    tn = test.select(component="N")[0]
    te = test.select(component="E")[0]
    windows = trigWindow(tz,winlen)
    paz = polarization(test,winlen)
    azstd = azimuthStd(paz["azimuth"],10)
    incicol = where(paz['incidence']<25,'r','k')
    azicol = where(azstd<15,'r','k')
    sNoise = surfaceNoise(paz['incidence'],azstd)

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
    offset = tz.data.std()
    ax[0].plot(tz.times("matplotlib"),tz.data+10*offset, color='red',linewidth=0.5, label="Z")
    ax[0].plot(tn.times("matplotlib"),tn.data+5*offset, color='green',linewidth=0.5, label="N")
    ax[0].plot(te.times("matplotlib"),te.data, color='blue',linewidth=0.5, label="E")
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
    offset = tz.data.std()
    
    tz.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    tn.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    te.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    ax[1].plot(tz.times("matplotlib"),tz.data+10*offset, color='red',linewidth=0.5, label="Z")
    ax[1].plot(tn.times("matplotlib"),tn.data+5*offset, color='green',linewidth=0.5, label="N")
    ax[1].plot(te.times("matplotlib"),te.data, color='blue',linewidth=0.5, label="E")
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

@click.command()
@click.argument('filename', type=click.Path(exists=True),nargs=-1)
@click.option('--calibration','-c')
@click.option('--zfactor', '-z', type=float)
@click.option('--nfactor', '-n', type=float)
@click.option('--efactor', '-e', type=float)
@click.option('--export/--no-export', default=False)
def main(filename,calibration,zfactor,nfactor,efactor,export):
    st = combine(filename)
    isexport = False
    if export: isexport=True
    if not calibration and not zfactor and not nfactor and not efactor:
        plot(st, filename)
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
        plot(st, filename, z, n, e)
if __name__ == '__main__':
    main()