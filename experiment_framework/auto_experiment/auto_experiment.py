"""Auto experiment module for the experiment framework.

This module provide template classes for automatic experiments. By implementing
experiment interfrace, the user can easily create their own experiments and run
batch experiments.

"""

from abc import ABC, abstractmethod
import tqdm


class AutoExperiment(ABC):
    """
    This is the abstract class for automatic experiments.
    """

    @abstractmethod
    def __init__(self):
        """
        This is the constructor of the class.
        """

    @abstractmethod
    def run(self):
        """
        This is the method that runs the experiment.
        """
    
    @abstractmethod
    def evaluate(self):
        """
        This is the method that evaluates the experiment.
        """

class ExperimentInterface(ABC):
    """
    This is the abstract class for the experiment interface.
    """

    @abstractmethod
    def load_parameters(self):
        """
        This is the method that parameterizes the experiment.
        """

    @abstractmethod
    def load_dataset(self):
        """
        This is the method that loads the dataset.
        """

    @abstractmethod
    def execute_experiment_process(self, parameters, dataset):
        """
        This is the method that processes the experiment.
        """
    
    @abstractmethod
    def summarize_results(self):
        """
        This is the method that summarizes the results.
        """

# Derived classes
class SimpleBatchExperiment(AutoExperiment):
    """
    This is the class for simple batch experiments.
    """

    def __init__(self, experiment_interface: ExperimentInterface, batch: int) -> None:
        """
        This is the constructor of the class.
        """
        self.experiment_interface = experiment_interface
        self.batch = batch
        self.parameter_group = self.experiment_interface.load_parameters()
        self.dataset = self.experiment_interface.load_dataset()

    def run(self):
        """
        Batch run the experiment.
        """
        progress_bar = tqdm.tqdm(total=len(self.parameter_group) * self.batch)
        for parameters in self.parameter_group:
            for _ in range(self.batch):
                self.experiment_interface.execute_experiment_process(parameters, self.dataset)
                progress_bar.update(1)
    
    def evaluate(self):
        """
        Evaluate the experiment.
        """
        self.experiment_interface.summarize_results()
