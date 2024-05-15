import re
import matplotlib.pyplot as plt

# Parse the text file and extract data
epochs = []
avg_disc_losses = []
avg_gen_losses = []
avg_gen_disc_accs = []
avg_disc_real_accs = []
avg_disc_fake_accs = []

with open("metrics.txt", "r") as file:
    for line in file:
        match_epoch = re.match(r'epoch (\d+)', line)
        match_avg_disc_loss = re.search(r'disc_loss (\d+\.\d+)', line)
        match_avg_gen_loss = re.search(r'gen_loss (\d+\.\d+)', line)
        match_avg_gen_disc_acc = re.search(r'gen_disc_acc (\d+\.\d+)', line)
        match_avg_disc_real_acc = re.search(r'disc_real_acc (\d+\.\d+)', line)
        match_avg_disc_fake_acc = re.search(r'disc_fake_acc (\d+\.\d+)', line)
        
        if match_epoch:
            epoch = int(match_epoch.group(1))
            epochs.append(epoch)
        if match_avg_disc_loss:
            avg_disc_loss = float(match_avg_disc_loss.group(1))
            avg_disc_losses.append(avg_disc_loss)
        if match_avg_gen_loss:
            avg_gen_loss = float(match_avg_gen_loss.group(1))
            avg_gen_losses.append(avg_gen_loss)
        if match_avg_gen_disc_acc:
            avg_gen_disc_acc = float(match_avg_gen_disc_acc.group(1))
            avg_gen_disc_accs.append(avg_gen_disc_acc)
        if match_avg_disc_real_acc:
            avg_disc_real_acc = float(match_avg_disc_real_acc.group(1))
            avg_disc_real_accs.append(avg_disc_real_acc)
        if match_avg_disc_fake_acc:
            avg_disc_fake_acc = float(match_avg_disc_fake_acc.group(1))
            avg_disc_fake_accs.append(avg_disc_fake_acc)

# Plot the graphs
plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, avg_disc_losses, label="Average Discriminator Loss")
# plt.plot(epochs, avg_gen_losses, label="Average Generator Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Average Discriminator and Generator Loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, avg_disc_real_accs, label="Average Discriminator Real Accuracy")
# plt.plot(epochs, avg_disc_fake_accs, label="Average Discriminator Fake Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Average Accuracy")
# plt.legend()

# plt.subplot(2, 1, 1)
plt.plot(epochs, avg_gen_disc_accs, label="Average Generator Discriminator Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Average Generator Discriminator Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
