def parse_data(data, outpath):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import datetime

    df = data['df']
    world_size = data['world_size']
    plt.figure(figsize=(9,8/5*world_size))


    for proc in range(0, world_size):
        X = df[df['rank'] == proc]
        if proc == 0:
            ax0 = plt.subplot(world_size, 1, proc+1)
        else:
            ax = plt.subplot(world_size, 1, proc+1, sharex=ax0)
        
        plt.stackplot(X['starttime'].values/60.0,
            X['time_comp'].values,
            X['time_send'].values,
            X['time_recv'].values,
            X['time_idle'].values,
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

    # plt.figure()
    # plt.hist(np.sum(times[:, 3:], axis=1), 500);
    # plt.savefig(os.path.join(outpath, "fig02.png"), format='png', dpi=300)

    # plt.figure()
    # plt.hist(times[np.sum(times[:, 3:], axis=1) >= 0.012, 6], 500);
    # plt.savefig(os.path.join(outpath, "fig03.png"), format='png', dpi=300)
    # print("mean = {}".format(np.mean(times[np.sum(times[:, 3:], axis=1) >= 0.012, 6])))

    # runtime
    total_seconds = df['endtime'].max() - df['starttime'].min()
    hours, rem = divmod(total_seconds, 60*60)
    minutes, rem = divmod(rem, 60)
    seconds, rem = divmod(rem, 1)
    data['runtime'] = "{:02}:{:02}:{:02}.{:02}".format(int(hours),
        int(minutes), int(seconds), int(rem*100))

    # starttime
    starttime = datetime.datetime.fromtimestamp(data['unix_timestamp_start'])
    data["starttime"] = starttime.strftime('%B %-d, %Y  %H:%M:%S')
