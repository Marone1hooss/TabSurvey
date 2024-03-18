import pyRAPL
import numpy as np

class EnergyTracker:
    def __init__(self):
        self.energy_values = []
        pyRAPL.setup()
        self.rapl = pyRAPL.Measurement('bar')

    def start_tracking(self):
        self.rapl.begin()

    def end_tracking(self):
        self.rapl.end()
        energy_consumed = self.rapl.result.pkg[0]
        
        self.energy_values.append(energy_consumed)

    def get_average_energy(self):
        if len(self.energy_values) == 0:
            return 0
        return np.mean(self.energy_values)

    def get_total_energy(self):
        return np.sum(np.array(self.energy_values))

""" # Example usage
if __name__ == "__main__":
    tracker = EnergyTracker()

    # Start energy tracking
    tracker.start_tracking()
    # Your code here
    for _ in range(1000000):  # Simulate some operations
        pass
    tracker.end_tracking()

    avg_energy = tracker.get_average_energy()
    total_energy = tracker.get_total_energy()

    print(f"Average Energy Consumed: {avg_energy} Joules")
    print(f"Total Energy Consumed: {total_energy} Joules") """
