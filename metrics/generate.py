import re

# Parse the text file and calculate average metrics for every 25 epochs
epochs_data = {}
current_epoch = None

with open("averaged_metrics.txt", "r") as file:
    for line in file:
        match_epoch = re.match(r'epoch (\d+)', line)
        match_disc_loss = re.search(r'disc_loss (\d+\.\d+)', line)
        match_gen_loss = re.search(r'gen_loss (\d+\.\d+)', line)
        match_gen_disc_acc = re.search(r'gen_disc_acc (\d+\.\d+)', line)
        match_disc_real_acc = re.search(r'disc_real_acc (\d+\.\d+)', line)
        match_disc_fake_acc = re.search(r'disc_fake_acc (\d+\.\d+)', line)
        
        if match_epoch:
            current_epoch = int(match_epoch.group(1))
            if current_epoch % 10 == 0:
                epochs_data[current_epoch] = {
                    'disc_loss': [],
                    'gen_loss': [],
                    'gen_disc_acc': [],
                    'disc_real_acc': [],
                    'disc_fake_acc': []
                }
        if current_epoch in epochs_data:
            if match_disc_loss:
                epochs_data[current_epoch]['disc_loss'].append(float(match_disc_loss.group(1)))
            if match_gen_loss:
                epochs_data[current_epoch]['gen_loss'].append(float(match_gen_loss.group(1)))
            if match_gen_disc_acc:
                epochs_data[current_epoch]['gen_disc_acc'].append(float(match_gen_disc_acc.group(1)))
            if match_disc_real_acc:
                epochs_data[current_epoch]['disc_real_acc'].append(float(match_disc_real_acc.group(1)))
            if match_disc_fake_acc:
                epochs_data[current_epoch]['disc_fake_acc'].append(float(match_disc_fake_acc.group(1)))

# Calculate average metrics for every 25 epochs and write to a file
with open("averaged_metrics_10_epochs.txt", "w") as output_file:
    for epoch, data in epochs_data.items():
        avg_disc_loss = sum(data['disc_loss']) / len(data['disc_loss'])
        avg_gen_loss = sum(data['gen_loss']) / len(data['gen_loss'])
        avg_gen_disc_acc = sum(data['gen_disc_acc']) / len(data['gen_disc_acc'])
        avg_disc_real_acc = sum(data['disc_real_acc']) / len(data['disc_real_acc'])
        avg_disc_fake_acc = sum(data['disc_fake_acc']) / len(data['disc_fake_acc'])
        
        output_file.write(f"epoch {epoch}, avg_disc_loss {avg_disc_loss}, avg_gen_loss {avg_gen_loss}\n")
        output_file.write(f"avg_gen_disc_acc {avg_gen_disc_acc}, avg_disc_real_acc {avg_disc_real_acc}, avg_disc_fake_acc {avg_disc_fake_acc}\n")
        output_file.write("\n")
