import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('run_26_train_correctness.csv')
val = pd.read_csv('run_26_val_correctness.csv')

plt.figure()
plt.plot(train['Step'], train['train/correctness'], label='Train')
plt.xlabel('Step')
plt.ylabel('Correctness')
plt.legend()
plt.tight_layout()
plt.savefig('train_correctness.png')

plt.figure()
plt.plot(val['Step'], val['val/correctness'], label='Validation')
plt.axhline(y=0.8, color='red', linestyle=':', linewidth=1.5, label='GPT 4.1')
plt.xlabel('Step')
plt.ylabel('Correctness')
plt.legend()
plt.tight_layout()
plt.savefig('val_correctness.png')

plt.close('all')


