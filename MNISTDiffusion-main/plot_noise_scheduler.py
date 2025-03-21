import numpy as np
import matplotlib.pyplot as plt

# Redefine the functions without relying on torch for compatibility
def cosine_schedule_np(start, end, timesteps, tau, epsilon):
    steps = np.linspace(0, timesteps, num=timesteps + 1, dtype=np.float32)
    steps = steps * (end - start) + start
    cos_value = np.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * np.pi * 0.5)
    cos_value = np.clip(cos_value, 0, 1)  # Ensure non-negative for exponentiation
    f_t = cos_value ** (2 * tau)
    betas = np.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
    
    return betas

def sigmoid_schedule_np(start=-3, end=3, timesteps=1000, tau=1.0, s=0):
    
    steps = np.linspace(0, timesteps, num=timesteps + 1, dtype=np.float32)
    # Normalize steps to the range [start, end]
    steps = steps * (end - start) + start
    # Compute the sigmoid-based function
    f_t = 1 / (1 + np.exp(-(((timesteps - steps) / timesteps)/ tau + s)))
    # Compute betas based on the schedule
    betas = np.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
    
    return betas 

# Generate data for plotting
timesteps = 1000
cosine_betas_np = cosine_schedule_np(start=0, end=1, timesteps=timesteps, tau=1.0, epsilon=0.008)
cosine_betas_np2 = cosine_schedule_np(start=0.1, end=1, timesteps=timesteps, tau=1.0, epsilon=0.008)
# sigmoid_betas_np = sigmoid_schedule_np(timesteps=timesteps, start=-4, end=3, tau=0.3, s=-0.5)

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(range(timesteps), cosine_betas_np, label="Cosine Schedule", color="blue")
plt.plot(range(timesteps), cosine_betas_np2, label="Cosine Schedule 2", color="blue")
# plt.plot(range(timesteps), sigmoid_betas_np, label="Sigmoid Schedule", color="orange")
plt.xlabel("Timesteps $t$", fontsize=20)
plt.ylabel("Noise scheduler $q_t$", fontsize=20)
# plt.title("Comparison of Cosine and Sigmoid Schedules")
plt.xticks(fontsize=18)  # Enlarged font size for x-axis ticks
plt.yticks(fontsize=18)  # Enlarged font size for y-axis ticks
# plt.legend()
plt.grid()

# Save the figure
plt.savefig("plots/schedule_comparison.png", dpi=300)

# Show the plot
plt.show()
