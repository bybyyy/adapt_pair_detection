import sys
import matplotlib
import matplotlib.pyplot as plt
from EventDataset import EventDataset

WLS_FAST_COUNT = 75 * 2 * 4
WLS_SLOW_COUNT = 75 * 2 * 4
EDGE_COUNT = 24
CAL_COUNT = 24

PAIR_COLOR = 'tab:orange'
NON_PAIR_COLOR = 'tab:blue'


def summarize_event_features(features):
    fast = features[:, :WLS_FAST_COUNT]
    slow = features[:, WLS_FAST_COUNT:WLS_FAST_COUNT + WLS_SLOW_COUNT]
    edge = features[:, WLS_FAST_COUNT + WLS_SLOW_COUNT:WLS_FAST_COUNT + WLS_SLOW_COUNT + EDGE_COUNT]
    cal = features[:,  WLS_FAST_COUNT + WLS_SLOW_COUNT + EDGE_COUNT :WLS_FAST_COUNT + WLS_SLOW_COUNT + EDGE_COUNT + CAL_COUNT]

    return {
        'WLS Fast Total Signal': fast.sum(axis=1),
        'WLS Slow Total Signal': slow.sum(axis=1),
        'Edge Detector Total Signal': edge.sum(axis=1),
        'Calorimeter Total Signal': cal.sum(axis=1),
        'Total Signal': features.sum(axis=1),
        'WLS Fast Active Channels': (fast > 0).sum(axis=1),
        'WLS Slow Active Channels': (slow > 0).sum(axis=1),
    }


def plot_histograms(statistics, labels):
    pair_mask = labels == 1
    nonpair_mask = labels == 0
    keys = list(statistics.keys())

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        ax.hist(statistics[key][nonpair_mask], bins=40, alpha=0.6, label='non-pair', color=NON_PAIR_COLOR, log=True)
        ax.hist(statistics[key][pair_mask], bins=40, alpha=0.6, label='pair', color=PAIR_COLOR, log=True)
        ax.set_title(key)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    plt.show()

def plot_event_scatter(features, labels):
    # pick a couple random events - 5 pair and 5 non-pair

    fig, axes = plt.subplots(2)

    n = 1
    event = features[n][WLS_FAST_COUNT:WLS_FAST_COUNT+WLS_SLOW_COUNT]
    event_type = "PAIR" if labels[n] == 1 else "NOT-PAIR"
    fig.suptitle(f"Event {n} - {event_type}")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(event))
    colorizer = matplotlib.colorizer.Colorizer(norm=norm, cmap="plasma")
    fig.colorbar(matplotlib.colorizer.ColorizingArtist(colorizer),
             ax=axes[1], orientation='horizontal', label='Some Units')
    
    for i in range(2):
        axes[i].set_yticks(range(4))
        axes[i].set_ylim(-.5,3.5)

    i = 0
    for z in range(4):
        for x in range(75):
            if event[i] != 0:
                axes[0].scatter(x, z, c=event[i], cmap="plasma", vmin=0, vmax=max(event))
            i += 1
        i+=75
    axes[0].set_title("X direction")

    i = 0
    for z in range(4):
        i+=75
        for y in range(75):
            if event[i] != 0:
                axes[1].scatter(y, z, c=event[i], cmap="plasma", vmin=0, vmax=max(event))
            i += 1
    # axes[1].yticks(range(4))
    axes[1].set_title("Y direction")

    plt.show()

def main():
    if (len(sys.argv) < 2):
        print("Usage: py plot_data.py datafile")
        exit()

    dataset = EventDataset(sys.argv[1])
    features = dataset.features_tensor.numpy()
    labels = dataset.labels_tensor.numpy().astype(int)

    statistics = summarize_event_features(features)
    plot_histograms(statistics, labels)

    plot_event_scatter(features, labels)


if __name__ == '__main__':
    main()
