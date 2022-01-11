from datetime import date
import enum
from obspy import read
from obspy.signal.invsim import evalresp_for_frequencies
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.signal.polarization import polarization_analysis
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stream
from matplotlib.dates import date2num
import matplotlib.pyplot as plt 
import matplotlib.dates as md
from numpy import record, where, array, median
import click
from os.path import basename

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

def trigWindow(tr, winlen=30, sta=5, lta=10, head=1.3, tail=0.7):
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

def polarization(st,winlen=30):
    if not streamCheck:
        return None
    print("Begin Polarization Analysis")
    result = polarization_analysis(st,winlen,1,0,10,st[0].stats.starttime+600,st[0].stats.endtime-600,False,'flinn')
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


def plot(test, filename, winlen=20):
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
    fig, ax = plt.subplots(7,sharex=True)
    fig.set_size_inches(8, 10)
    # fig.suptitle('{}'.format(str(tz).replace('|','\n')), x=0.1,ha="left", fontsize=12)
    efectiveStart = tz.stats.starttime + 600
    efectiveEnd = tz.stats.endtime - 600
    offset = tz.data.std()
    ax[0].plot(tz.times("matplotlib"),tz.data+10*offset, color='red',linewidth=0.5, label="Z")
    ax[0].plot(tn.times("matplotlib"),tn.data+5*offset, color='green',linewidth=0.5, label="N")
    ax[0].plot(te.times("matplotlib"),te.data, color='blue',linewidth=0.5, label="E")
    ax[0].legend()

    tz.trim(efectiveStart,efectiveEnd)
    tn.trim(efectiveStart,efectiveEnd)
    te.trim(efectiveStart,efectiveEnd)
    ax[0].set_ylim(min(te.data),max(tz.data)+10*offset)
    offset = tz.data.std()
    
    tz.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    tn.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    te.filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2, zerophase=True)
    ax[1].plot(tz.times("matplotlib"),tz.data+10*offset, color='red',linewidth=0.5, label="Z")
    ax[1].plot(tn.times("matplotlib"),tn.data+5*offset, color='green',linewidth=0.5, label="N")
    ax[1].plot(te.times("matplotlib"),te.data, color='blue',linewidth=0.5, label="E")
    ax[1].legend()
    
    windowNoSurface = 0
    for i, w in enumerate(windows):
        windowNoSurface += 1 if sNoise[i] == 'k' else 0
        ax[1].axvspan(w[0].matplotlib_date, w[1].matplotlib_date, alpha=0.5, color=sNoise[i])
    print("{} windows with surface noise detected, removing windows".format(len(windows)-windowNoSurface))
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

    # ax[2].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()
    ax[1].yaxis.tick_right()
    ax[2].yaxis.tick_right()
    ax[3].yaxis.tick_right()
    ax[4].yaxis.tick_right()
    ax[5].yaxis.tick_right()
    ax[6].yaxis.tick_right()

    # ax[0].xaxis.tick_top()
    # ax[1].xaxis.tick_top()
    # ax[2].xaxis.tick_top()
    # ax[3].xaxis.tick_top()
    # ax[4].xaxis.tick_top()
    # ax[5].xaxis.tick_top()

    # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    # ax[0].xaxis.set_major_formatter(xfmt)

    ax[0].xaxis.set_major_formatter(
        md.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
    
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
    noiseLevel = []
    noiseLevel.append(date2num(efectiveStart+1200))
    
    startBar = date2num(efectiveEnd)
    setupNoise = date2num(efectiveEnd-1200)
    transientNoise = date2num(efectiveEnd-(recordingLength-20-transientLength)*60)
    incidenceNoise = 0
    azimuthNoise = 0
    for i in range(len(incicol)):
        incidenceNoise += 1 if incicol[i]=='r' else 0
        azimuthNoise += 1 if azicol[i]=='r' else 0
    incidenceNoise *= winlen
    azimuthNoise *= winlen
    incidenceNoise = date2num(efectiveEnd-incidenceNoise)
    azimuthNoise = date2num(efectiveEnd-azimuthNoise)
    ax[6].set_yticks([0,-1,-2,-3], labels=noiseSource)
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
    for tr in st:
        latestStart = tr.stats.starttime if latestStart < tr.stats.starttime else latestStart
        earliestend = tr.stats.endtime if earliestend > tr.stats.endtime else earliestend
    st.trim(latestStart, earliestend)
    return st

@click.command()
@click.argument('filename', type=click.Path(exists=True),nargs=-1)
def main(filename):
    st = combine(filename)
    plot(st, filename)

if __name__ == '__main__':
    main()