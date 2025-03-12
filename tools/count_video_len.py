import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

# Load the data from the JSONL file
types = ['loss_only', 'info_only', 'loss+info']
def count_video_lengths(file_path):
    video_length_cnt = {}

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for boundary in data['boundaries']:
                video_length = boundary[1] - boundary[0] + 1
                video_length_cnt[video_length] = video_length_cnt.get(video_length, 0) + 1

    return video_length_cnt

for type in types:
    file_path = f'./result/{type}.jsonl'
    video_length_cnt = count_video_lengths(file_path)
    print(max(video_length_cnt.keys()))

    # Calculate the mean length
    total_length = sum([length * cnt for length, cnt in video_length_cnt.items()])
    total_cnt = sum(video_length_cnt.values())
    print(f'Total length: {total_length}, Total count: {total_cnt}')
    mean_length = total_length / total_cnt
    print(f'Mean length: {mean_length}')

    # Calculate the frequency of each bin
    bins = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    bins.append(max(video_length_cnt.keys()))
    bin_labels = [str(bins[i] + 1) if bins[i + 1] - bins[i] == 1 else str(bins[i] + 1) + '~' + str(bins[i + 1]) for i in range(len(bins) - 2)]
    bin_labels.append(str(bins[-2] + 1) + '+')
    bin_counts = [0] * (len(bins) - 1)

    for length, cnt in video_length_cnt.items():
        for i in range(len(bins) - 1):
            if bins[i] < length <= bins[i + 1]:
                bin_counts[i] += cnt
                break

    # normalize the bin_counts
    bin_counts = [cnt / total_cnt for cnt in bin_counts]

    plt.clf()

    # Plot the bar chart
    plt.bar(bin_labels, bin_counts, edgecolor='black')

    plt.xlabel('Length (frames)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20, labelpad=15)  # Add padding between ylabel and ytick
    plt.xticks(rotation=45, ha='right', fontsize=16)  # Rotate x-axis labels to prevent overlap
    plt.yticks(fontsize=16)  # Set y-axis labels font size
    plt.tight_layout()  # Adjust layout to make room for the rotated labels
    output_path = f'./result/video_length_distribution_{type}.png'
    plt.savefig(output_path, dpi=300)