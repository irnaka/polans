from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.signal.polarization import polarization_analysis
from obspy.core.utcdatetime import UTCDateTime
from matplotlib.dates import date2num
import matplotlib.pyplot as plt 
import matplotlib.dates as md
from numpy import where
import click

def makeWindow(tr,winlen=30):
    startwindow = tr.stats.starttime
    endwindow = startwindow + winlen
    windowlist = []
    while(endwindow < tr.stats.endtime):
        windowlist.append((startwindow, endwindow))
        startwindow += winlen
        endwindow += winlen
    return windowlist

def trigWindow(tr, winlen=30, sta=5, lta=10, head=1.3, tail=0.7):
    wd = makeWindow(tr, winlen)
    df = tr.stats.sampling_rate
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
    for x in badwindow:
        wd.pop(x)
    return wd

def streamCheck(st):
    if test.select(component="Z") and test.select(component="N") and test.select(component="E"):
        return True
    else:
        return False

def polarization(st):
    if not streamCheck:
        return None
    result = polarization_analysis(st,10,0.5,0,10,st[0].stats.starttime,st[0].stats.endtime,False,'flinn')
    return result

def plot(pathfile):
    test = read(pathfile)
    tz = test.select(component="Z")[0]
    windows = trigWindow(tz)
    paz = polarization(test)

    incicol = where(paz['incidence']<30,'k','k')
    rectcol = where(paz['rectilinearity']>0.7,'k','k')
    fig, ax = plt.subplots(5,sharex=True)
    fig.set_size_inches(8, 10)
    fig.suptitle('{}'.format(str(tz).replace('|','\n')), fontsize=12)
    ax[0].plot(tz.times("matplotlib"),tz.data, color='black',linewidth=0.5)
    for w in windows:
        ax[0].axvspan(w[0].matplotlib_date, w[1].matplotlib_date, alpha=0.5, color='gray')
    ax[1].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['azimuth'], color='black',s=1)
    ax[2].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['incidence'], color=incicol,s=1)
    ax[3].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['rectilinearity'], color=rectcol,s=1)
    ax[4].scatter( [date2num(UTCDateTime(t)) for t in paz['timestamp']],paz['planarity'], color=rectcol,s=1)

    ax[1].set_ylim(-20,200)
    ax[2].set_ylim(-10,100)
    ax[3].set_ylim(-0.1,1.1)
    ax[4].set_ylim(-0.1,1.1)

    ax[1].set_yticks((0,90,180))
    ax[2].set_yticks((0,45,90))
    ax[3].set_yticks((0,0.5,1))
    ax[4].set_yticks((0,0.5,1))

    ax[1].set_ylabel('Azimuth')
    ax[2].set_ylabel('Incidence')
    ax[3].set_ylabel('Rectilinearity')
    ax[4].set_ylabel('Planarity')

    # ax[1].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()
    ax[1].yaxis.tick_right()
    ax[2].yaxis.tick_right()
    ax[3].yaxis.tick_right()
    ax[4].yaxis.tick_right()

    # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    # ax[0].xaxis.set_major_formatter(xfmt)

    ax[0].xaxis.set_major_formatter(
        md.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))

    plt.subplots_adjust(hspace=.0)
    plt.savefig('result.png')
    # plt.show()


@click.command()
@click.argument('filename', type=click.Path(exists=True))
def main(filename):
    plot(filename)

if __name__ == '__main__':
    main()