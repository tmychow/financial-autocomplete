import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('run_26_train_correctness.csv')
val = pd.read_csv('run_26_val_correctness.csv')

plt.plot(train['Step'], train['train/correctness'], label='Train')
plt.plot(val['Step'], val['val/correctness'], label='Val')
plt.xlabel('Step')
plt.ylabel('Correctness')
plt.legend()
plt.tight_layout()
plt.savefig('correctness.png')


