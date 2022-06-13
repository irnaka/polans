from datetime import date
from re import L
from obspy import read
from obspy.signal.cross_correlation import correlate,xcorr_max
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
# from obspy.signal.polarization import polarization_analysis
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stream
from obspy.signal.filter import bandpass
from obspy.signal.invsim import cosine_taper
from matplotlib.dates import date2num
import matplotlib.pyplot as plt 
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
from numpy import cos, sin, angle, where, array, median, abs, fft, argsort, rint, sqrt, real;
from numpy import rad2deg
import click
from os.path import basename,dirname
from os import sep
import os
import numpy as np
import obspy
import math

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

def flinn_modified(stream, noise_thres=0):
    mask = (stream[0][:] ** 2 + stream[1][:] ** 2 + stream[2][:] ** 2
            ) > noise_thres
    x = np.zeros((3, mask.sum()), dtype=np.float64)
    # East
    x[0, :] = stream[2][mask]
    # North
    x[1, :] = stream[1][mask]
    # Z
    x[2, :] = stream[0][mask]

    covmat = np.cov(x)
    eigvec, eigenval, v = np.linalg.svd(covmat)
    # Rectilinearity defined after Montalbetti & Kanasewich, 1970
    rect = 1.0 - sqrt(eigenval[1] / eigenval[0])
    # Planarity defined after [Jurkevics1988]_
    plan = 1.0 - (2.0 * eigenval[2] / (eigenval[1] + eigenval[0]))
    azimuth = rad2deg(math.atan2(eigvec[0][0], eigvec[1][0]))
    eve = sqrt(eigvec[0][0] ** 2 + eigvec[1][0] ** 2)
    incidence = rad2deg(math.atan2(eve, eigvec[2][0]))
    if azimuth < 0.0:
        azimuth = 360.0 + azimuth
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if azimuth > 180.0:
            azimuth -= 180.0
        else:
            azimuth += 180.0
    return azimuth, incidence, rect, plan

def _get_s_point(stream, stime, etime):
    """
    Function for computing the trace dependent start time in samples

    :param stime: time to start
    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: time to end
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :returns: spoint, epoint
    """
    slatest = stream[0].stats.starttime
    eearliest = stream[0].stats.endtime
    for tr in stream:
        if tr.stats.starttime >= slatest:
            slatest = tr.stats.starttime
        if tr.stats.endtime <= eearliest:
            eearliest = tr.stats.endtime

    nostat = len(stream)
    spoint = np.empty(nostat, dtype=np.int32)
    epoint = np.empty(nostat, dtype=np.int32)
    # now we have to adjust to the beginning of real start time
    if slatest > stime:
        msg = "Specified start time is before latest start time in stream"
        raise ValueError(msg)
    if eearliest < etime:
        msg = "Specified end time is after earliest end time in stream"
        raise ValueError(msg)
    for i in range(nostat):
        offset = int(((stime - slatest) / stream[i].stats.delta + 1.))
        negoffset = int(((eearliest - etime) / stream[i].stats.delta + 1.))
        diffstart = slatest - stream[i].stats.starttime
        frac, _ = math.modf(diffstart)
        spoint[i] = int(diffstart)
        if frac > stream[i].stats.delta * 0.25:
            msg = "Difference in start times exceeds 25% of sampling rate"
            warnings.warn(msg)
        spoint[i] += offset
        diffend = stream[i].stats.endtime - eearliest
        frac, _ = math.modf(diffend)
        epoint[i] = int(diffend)
        epoint[i] += negoffset

    return spoint, epoint

def polarization_analysis(stream, win_len, win_frac, frqlow, frqhigh, stime,
                          etime, verbose=False, method="pm", var_noise=0.0,
                          adaptive=True):
    """
    Method carrying out polarization analysis with the [Flinn1965b]_,
    [Jurkevics1988]_, ParticleMotion, or [Vidale1986]_ algorithm.

    :param stream: 3 component input data.
    :type stream: :class:`~obspy.core.stream.Stream`
    :param win_len: Sliding window length in seconds.
    :type win_len: float
    :param win_frac: Fraction of sliding window to use for step.
    :type win_frac: float
    :param var_noise: resembles a sphere of noise in PM where the 3C is
        excluded
    :type var_noise: float
    :param frqlow: lower frequency. Only used for ``method='vidale'``.
    :type frqlow: float
    :param frqhigh: higher frequency. Only used for ``method='vidale'``.
    :type frqhigh: float
    :param stime: Start time of interest
    :type stime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param etime: End time of interest
    :type etime: :class:`obspy.core.utcdatetime.UTCDateTime`
    :param method: the method to use. one of ``"pm"``, ``"flinn"`` or
        ``"vidale"``.
    :type method: str
    :param adaptive: switch for adaptive window estimation (defaults to
        ``True``). If set to ``False``, the window will be estimated as
        ``3 * max(1/(fhigh-flow), 1/flow)``.
    :type adaptive: bool
    :rtype: dict
    :returns: Dictionary with keys ``"timestamp"`` (POSIX timestamp, can be
        used to initialize :class:`~obspy.core.utcdatetime.UTCDateTime`
        objects), ``"azimuth"``, ``"incidence"`` (incidence angle) and
        additional keys depending on used method: ``"azimuth_error"`` and
        ``"incidence_error"`` (for method ``"pm"``), ``"rectilinearity"`` and
        ``"planarity"`` (for methods ``"flinn"`` and ``"vidale"``) and
        ``"ellipticity"`` (for method ``"flinn"``). Under each key a
        :class:`~numpy.ndarray` is stored, giving the respective values
        corresponding to the ``"timestamp"`` :class:`~numpy.ndarray`.
    """
    if method.lower() not in ["pm", "flinn", "vidale"]:
        msg = "Invalid method ('%s')" % method
        raise ValueError(msg)

    res = []

    if stream.get_gaps():
        msg = 'Input stream must not include gaps:\n' + str(stream)
        raise ValueError(msg)

    if len(stream) != 3:
        msg = 'Input stream expected to be three components:\n' + str(stream)
        raise ValueError(msg)

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = "sampling rates of traces in stream are not equal"
        raise ValueError(msg)

    if verbose:
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    spoint, _epoint = _get_s_point(stream, stime, etime)
    if method.lower() == "vidale":
        res = vidale_adapt(stream, var_noise, fs, frqlow, frqhigh, spoint,
                           stime, etime)
    else:
        nsamp = int(win_len * fs)
        nstep = int(nsamp * win_frac)
        newstart = stime
        tap = cosine_taper(nsamp, p=0.22)
        offset = 0
        while (newstart + (nsamp + nstep) / fs) < etime:
            try:
                for i, tr in enumerate(stream):
                    dat = tr.data[spoint[i] + offset:
                                  spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    if tr.stats.channel[-1].upper() == "Z":
                        z = dat.copy()
                    elif tr.stats.channel[-1].upper() == "N":
                        n = dat.copy()
                    elif tr.stats.channel[-1].upper() == "E":
                        e = dat.copy()
                    else:
                        msg = "Unexpected channel code '%s'" % tr.stats.channel
                        raise ValueError(msg)

                data = [z, n, e]
            except IndexError:
                break

            # we plot against the centre of the sliding window
            if method.lower() == "pm":
                azimuth, incidence, error_az, error_inc = \
                    particle_motion_odr(data, var_noise)
                res.append(np.array([newstart.timestamp + float(nstep) / fs,
                           azimuth, incidence, error_az, error_inc]))
            if method.lower() == "flinn":
                azimuth, incidence, reclin, plan = flinn_modified(data, var_noise)
                res.append(np.array([newstart.timestamp + float(nstep) / fs,
                                    azimuth, incidence, reclin, plan]))

            if verbose:
                print(newstart, newstart + nsamp / fs, res[-1][1:])
            offset += nstep

            newstart += float(nstep) / fs

    res = np.array(res)

    result_dict = {"timestamp": res[:, 0],
                   "azimuth": res[:, 1],
                   "incidence": res[:, 2]}
    if method.lower() == "pm":
        result_dict["azimuth_error"] = res[:, 3]
        result_dict["incidence_error"] = res[:, 4]
    elif method.lower() == "vidale":
        result_dict["rectilinearity"] = res[:, 3]
        result_dict["planarity"] = res[:, 4]
        result_dict["ellipticity"] = res[:, 5]
    elif method.lower() == "flinn":
        result_dict["rectilinearity"] = res[:, 3]
        result_dict["planarity"] = res[:, 4]
    return result_dict

def polarization(st,winlen=30):
    if not streamCheck:
        return None
    print("Begin Polarization Analysis")
    result = polarization_analysis(st,winlen,1,0,10,st[0].stats.starttime+600,st[0].stats.endtime-600,False,'flinn')
    for i,item in enumerate(result['incidence']):
       # result['incidence'][i] = rotate90(item) 
       result['incidence'][i] = 90-abs(item)
    # print([UTCDateTime(t) for t in result['timestamp']])
    return result

def azimuthRotate(inp, rotation):
    lst = []
    for i in range(len(inp)):
        val = inp[i]-rotation
        val = 180+val if val<0 else val
        lst.append(val)
    return array(lst)

def azimuthStd(inp360, wind):
    lst = []
    # wrap azimuth and back azimuth to avoid ambiguity
    inp = [x if x <= 180 else x-180 for x in inp360]
    # rotate azimuth by 90 degrees
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
        lst.append(min((np.percentile(wow,75)-np.percentile(wow,25))/2,(np.percentile(wow90,75)-np.percentile(wow90,25))/2))
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
    azstd = azimuthStd(paz["azimuth"],20)
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

    minmax_axis = [min(te.times("matplotlib")),max(te.times("matplotlib"))]
    # leg0 = ax[0].legend()

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
     
    tz.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    tn.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    te.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    offset = np.max([tz.data.std(),tn.data.std(),te.data.std()])
    ax[1].plot(te.times("matplotlib"),te.data*1.0, color='blue',linewidth=0.25, alpha=0.8, label="E")
    ax[1].plot(tn.times("matplotlib"),tn.data*1.0+5*offset, color='green',linewidth=0.25, alpha=0.8, label="N")
    ax[1].plot(tz.times("matplotlib"),tz.data*1.0+10*offset, color='red',linewidth=0.25, alpha=0.8, label="Z")
    leg1 = ax[1].legend()
    leg7 = ax[7].legend()
    
    # for legobj in leg0.legendHandles:
    #     legobj.set_linewidth(5.0)
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

    ax[2].text(max(te.times("matplotlib")),25,f"{min(paz['incidence']):05.2f}")
    ax[2].text(max(te.times("matplotlib")),55,f"{max(paz['incidence']):05.2f}")

    ax[3].text(max(te.times("matplotlib")),60,f"{min(paz['azimuth']):05.1f}")
    ax[3].text(max(te.times("matplotlib")),225,f"{max(paz['azimuth']):05.1f}")

    ax[4].text(max(te.times("matplotlib")),0.2,f"{min(paz['rectilinearity']):03.2f}")
    ax[4].text(max(te.times("matplotlib")),0.5,f"{max(paz['rectilinearity']):03.2f}")

    ax[5].text(max(te.times("matplotlib")),0.2,f"{min(paz['planarity']):03.2f}")
    ax[5].text(max(te.times("matplotlib")),0.5,f"{max(paz['planarity']):03.2f}")

    ax[0].set_xlim(minmax_axis[0],minmax_axis[1])

    ax[2].set_ylim(-10,100)
    ax[3].set_ylim(-20,380)
    ax[4].set_ylim(-0.1,1.1)
    ax[5].set_ylim(-0.1,1.1)
    ax[6].set_ylim(-4,1)

    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks((0,45,90))
    ax[3].set_yticks((0,90,180,270,360))
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
        if 'merged' in instrument.lower():
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

    if statistic_mode=="mean":
        print(f"|=================================================================================================|")
        print(f"|       (    (    (    (                   (    (        (                      )  (              |")
        print(f"|       )\ ) )\ ) )\ ) )\ )     (    (     )\ ) )\ )  (  )\ )   (      *   ) ( /(  )\ )           |")
        print(f"|      (()/((()/((()/((()/(     )\   )\   (()/((()/(( )\(()/(   )\   ` )  /( )\())(()/(           |")
        print(f"|       /(_))/(_))/(_))/(_))  (((_|(((_)(  /(_))/(_))((_)/(_)|(((_)(  ( )(_)|(_)\  /(_))          |")
        print(f"|      (_)) (_))_(_)) (_))    )\___)\ _ )\(_)) (_))((_)_(_))  )\ _ )\(_(_())  ((_)(_))            |")
        print(f"|      | |  | |_ | _ \/ __|  ((/ __(_)_\(_) |  |_ _|| _ ) _ \ (_)_\(_)_   _| / _ \| _ \           |")
        print(f"|      | |__| __||  _/\__ \   | (__ / _ \ | |__ | | | _ \   /  / _ \   | |  | (_) |   /           |")
        print(f"|      |____|_|  |_|  |___/    \___/_/ \_\|____|___||___/_|_\ /_/ \_\  |_|   \___/|_|_\           |")
        print(f"|=================================================================================================|")
    if statistic_mode=="mean":
        print("Calculating mean calibration factor ± standard deviation")
    elif statistic_mode=="median":
        print("Calculating median calibration factor ± 1.5*semi-interquartile range (SIR)")
    print(f"START TIME          : {starttime}")
    print(f"END TIME            : {endtime}")
    print(f"Calibration duration: {endtime-starttime} seconds")

    for i,instrument in enumerate(data):
        data[i].trim(starttime,endtime)
        data[i].filter("bandpass",freqmin=target_frequency[0],freqmax=target_frequency[1])

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

    correlation_coef = np.zeros((number_of_instrument,len(list_component)))
    suggested_shift = np.zeros((number_of_instrument,len(list_component)))
    for ilc,lc in enumerate(list_component): 
        # calibrating Z component, then N, then E
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
            # calculating correlation coefficient
            for ic,c in enumerate(list_component):
                if data[ii][ilc].stats.channel[-1] == c:
                    if np.isnan(calibratorid):
                        # compare it with the first data
                        _tmp1 = data[ii][ilc].data
                        for ic2,c2 in enumerate(list_component):
                            if data[0][ic2].stats.channel[-1] == c:
                                _tmp2 = data[0][ic2].data/np.max(np.abs(data[0][ic2].data))
                    else:
                        _tmp1 = data[ii][ilc].data/np.max(np.abs(data[ii][ilc].data))
                        for ic2,c2 in enumerate(list_component):
                            if data[calibratorid][ic2].stats.channel[-1] == c:
                                _tmp2 = data[calibratorid][ic2].data/np.max(np.abs(data[calibratorid][ic2].data))
                    _tmplen = np.min([len(_tmp1),len(_tmp2)])
                    correlation_coef[ii,ic] = np.corrcoef(_tmp1[:_tmplen],_tmp2[:_tmplen])[0,1]
                    suggested_shift[ii,ic] = xcorr_max(correlate(_tmp1[:_tmplen],_tmp2[:_tmplen],0))[0]

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

    print(f"|=================================================================================================")
    print(f"|ID|             Z             |             N             |             E             | Filename ")
    print(f"|  |     Cal     | Match |Shift|     Cal     | Match |Shift|     Cal     | Match |Shift|          ")
    print(f"|=================================================================================================")
    for ii in range(number_of_instrument):
        if statistic_mode=="mean":
            print(f"|{ii+1:02d}|{calibration_factor[ii,0]:6.4f}±{calibration_factor_std[ii,0]:6.4f}|{correlation_coef[ii,0]*100:6.2f}%|{int(np.floor(suggested_shift[ii,0])):5d}|{calibration_factor[ii,1]:6.4f}±{calibration_factor_std[ii,1]:6.4f}|{correlation_coef[ii,1]*100:6.2f}%|{int(np.floor(suggested_shift[ii,1])):5d}|{calibration_factor[ii,2]:6.4f}±{calibration_factor_std[ii,2]:6.4f}|{correlation_coef[ii,2]*100:6.2f}%|{int(np.floor(suggested_shift[ii,2])):5d}| {filename[ii]}")
        elif statistic_mode=="median":
            # calculating semi interquartile range
            _siqrz = (calibration_factor_q3[ii,0]-calibration_factor_q1[ii,0])*1.5/2
            _siqrn = (calibration_factor_q3[ii,1]-calibration_factor_q1[ii,1])*1.5/2
            _siqre = (calibration_factor_q3[ii,2]-calibration_factor_q1[ii,2])*1.5/2
            print(f"|{ii+1:02d}|{calibration_factor[ii,0]:6.4f}±{_siqrz:6.4f}|{correlation_coef[ii,0]*100:6.2f}%|{int(np.floor(suggested_shift[ii,0])):5d}|{calibration_factor[ii,1]:6.4f}±{_siqrn:6.4f}|{correlation_coef[ii,1]*100:6.2f}%|{int(np.floor(suggested_shift[ii,1])):5d}|{calibration_factor[ii,2]:6.4f}±{_siqre:6.4f}|{correlation_coef[ii,2]*100:6.2f}%|{int(np.floor(suggested_shift[ii,2])):5d}| {filename[ii]}")
    print(f"|=================================================================================================")

def merging1compto3compfiles(filelist):
    if len(filelist)!=3:
        raise IOError("The number of input files must be equal to 3!")
    st = Stream()
    for item in filelist:
        _tmp_st = read(item).merge(fill_value='latest')
        if len(_tmp_st)!=1:
            print(f"{item} has more than 1 component!")
            raise IOError("Input error!")
        else:
            st += _tmp_st.copy()
    latestStart = UTCDateTime(0)
    earliestend = UTCDateTime(2147483647)
    st.merge(fill_value='latest')
    if len(st)!=3:
        raise IOError("component duplicate is detected! Make sure that you give three single component files with different component.")
    for tr in st:
        latestStart = tr.stats.starttime if latestStart < tr.stats.starttime else latestStart
        earliestend = tr.stats.endtime if earliestend > tr.stats.endtime else earliestend
    st.trim(latestStart, earliestend)

    st.write("merged.mseed",format="MSEED")
    print("Data has been succesfully merged!")

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
@click.option('--incidence_threshold','-i', type=float, default=25.0)
@click.option('--azimuth_threshold','-a', type=float, default=20.0)
def main(filename,mode,calibration,zfactor,nfactor,efactor,export,calibrator,trigger_on,trigger_off,incidence_threshold,azimuth_threshold):
    if mode=="QC":
        st = combine(filename)
        isexport = False
        if export: isexport=True
        if not calibration and not zfactor and not nfactor and not efactor:
            plot(st, filename,  incth=incidence_threshold, azistdth=azimuth_threshold,trigger_on=trigger_on, trigger_off=trigger_off)
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
            
            plot(st, filename, z, n, e, incth=incidence_threshold, azistdth=azimuth_threshold, isexport=isexport, trigger_on=trigger_on, trigger_off=trigger_off)
    elif mode=="CALIBRATION":
        calibrateC(filename,calibrator=calibrator,target_frequency=[1,5],min_number_of_cycle=100,statistic_mode="mean")
        print()
        calibrateC(filename,calibrator=calibrator,target_frequency=[1,5],min_number_of_cycle=100,statistic_mode="median")
    elif mode=="MERGE":
        print("The program will combine several single component files into a single 3 component file!")
        print("Make sure that you only select 3 appropriate files!")
        merging1compto3compfiles(filename)
    else:
        print("Not yet implemented!")
if __name__ == '__main__':
    main()