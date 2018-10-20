def parse_data(data, outpath):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import datetime

    times = data['times']
    world_size = int(times[-1,0])+1

    plt.figure(figsize=(9,8/5*world_size))


    for proc in range(0, world_size):
        X = times[times[:, 0] == proc, :]
        if proc == 0:
            ax0 = plt.subplot(world_size, 1, proc+1)
        else:
            ax = plt.subplot(world_size, 1, proc+1, sharex=ax0)
        
        # t_tot = np.sum(X[:, 3:], axis=1)/100.0
        plt.stackplot(X[:, 1]/60,
            # X[:,3]/t_tot, X[:,4]/t_tot, X[:,5]/t_tot, X[:,6]/t_tot,
            X[:,3], X[:,4], X[:,5], X[:,6],
            labels=["Comp", "Send", "Recv", "Idle"])
        
        
        if proc == world_size-1:
            plt.legend()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.xlabel("Minutes")
            # plt.ylabel("Percent")
            plt.ylabel("Seconds")
        else:
            plt.axis("off")

    plt.savefig(os.path.join(outpath, "fig01.png"), format='png', dpi=300)

    # runtime
    total_seconds = np.amax(times[:, 2]) - np.amin(times[:, 1])
    hours, rem = divmod(total_seconds, 60*60)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    data['runtime'] = "{:02}:{:02}:{:02}.{:02}".format(int(hours),
        int(minutes), int(seconds), int(rem*100))

    # starttime
    starttime = datetime.datetime.fromtimestamp(data['unix_timestamp_start'])
    data["starttime"] = starttime.strftime('%B %-d, %Y  %H:%M:%S')
