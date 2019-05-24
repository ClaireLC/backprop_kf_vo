import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train_path', default=None, help='log/FILE_NAME')
parser.add_argument('--val', dest='val_path', default=None, help='log/FILE_NAME')
args = parser.parse_args()

# Read loss file
train_path = args.train_path
val_path = args.val_path

# Setup plot
fig, axs = plt.subplots()

if train_path is not None:

    ### Train ###

    # Extract data
    reader = csv.reader(open(train_path, newline=''), delimiter=',', quotechar='|')
    max_iter = 0
    next(reader)  # skip first element
    data = []
    for row in reader:
        if float(row[1])>max_iter and float(row[0])==0:
            max_iter = float(row[1])
        cur_time = float(row[0])*max_iter+float(row[1])
        temp = []
        temp.append(float(cur_time))
        # print(len(row))
        for i in range(len(row)):
            temp.append(float(row[i]))
        data.append(temp)
    data = np.array(data)

    # Extract average loss per epoch
    dataE = []
    for t in range(data.shape[0]):
        temp = 0
        if data[t,2]<=max_iter:
            temp += 1/max_iter*data[t,3]
        if data[t,2]==max_iter:
            dataE.append(temp)
    dataE = np.array(dataE)

    # Plot training loss
    # axs.plot(data[:,0], data[:,3])
    axs.plot(range(dataE.shape[0]), dataE,'.-b')
    # axs[0].set_xlim(0, time_ind)

    # Print statistical results
    print('training mean RMS error: {}'.format(np.mean(np.sqrt(data[:,2]))))
    print('training std  RMS error: {}'.format(np.std(np.sqrt(data[:,2]))))

if val_path is not None:
    ### Validation ###

    reader = csv.reader(open(val_path, newline=''), delimiter=',', quotechar='|')
    next(reader)
    val_loss = []
    for row in reader:
        val_loss.append(float(row[1].strip())) # val loss

    # Plot validation loss
    val_loss = np.array(val_loss)
    axs.plot(range(val_loss.shape[0]), val_loss, '.-r')

    # Plot labels
    axs.set_xlabel('epoch')
    axs.set_ylabel('MSE training loss')
    axs.grid(True)
    fig.tight_layout()

if val_path is not None or train_path is not None:
    # Display and save image
    if train_path:
        fig_name = train_path[:-9] + '.png'
    else:
        fig_name = val_path[:-13] + '.png'
    plt.savefig(fig_name, format="png")
    #plt.show()
