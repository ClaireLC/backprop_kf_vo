import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Read loss file
start_time = '1557525033'
file_path = 'claire_models/'+start_time+'_loss.txt'

# Extract data
reader = csv.reader(open(file_path, newline=''), delimiter=',', quotechar='|')
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

# Plot loss
fig, axs = plt.subplots()
# axs.plot(data[:,0], data[:,3])
axs.plot(range(dataE.shape[0]), dataE,'.')
# axs[0].set_xlim(0, time_ind)
axs.set_xlabel('epoch')
axs.set_ylabel('MSE training loss')
axs.grid(True)
fig.tight_layout()

# Print statistical results
print('training mean RMS error: {}'.format(np.mean(np.sqrt(data[:,2]))))
print('training std  RMS error: {}'.format(np.std(np.sqrt(data[:,2]))))

# Display and save image
fig_name = 'claire_models/'+start_time+'_loss.png'
plt.savefig(fig_name, format="png")
#plt.show()
