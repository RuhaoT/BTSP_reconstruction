"""Temp file for dataset generation and encoding

TODO(Ruhao Tian): Refactor this file with pyarrow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline
from PIL import Image
import pyarrow as pa
import pyarrow.csv as pc


class MinimalBTSPDataset:
    """A minimal multi-modal dataset for the BTSP project."""

    def __init__(self, coordinate_precision: int = 8, color_precision: int = 8):
        self.input_file = None
        self.precise_raw_data: pa.Table = None
        self.control_points_normalized = None
        self.coordinate_precision = coordinate_precision
        self.color_precision = color_precision
        self.binary_tensors: np.ndarray = None
        self.binary_mask: np.ndarray = None
        self.file_schema = ["x", "y", "r", "g", "b"]
        self.mask_types = ["coordinates", "colors", "full", "none"]

    def from_file(self, input_file: str):
        """Load the dataset from a CSV file."""
        self.input_file = input_file
        self.precise_raw_data = pc.read_csv(input_file)

        # check the schema of the input file
        if self.precise_raw_data.column_names != self.file_schema:
            raise ValueError(
                (
                    "Input file schema is incorrect. Expected:"
                    f"{self.file_schema}"
                    ", got:"
                    f"{self.precise_raw_data.columns.tolist()}"
                )
            )

        return self

    def from_binary_tensors(self, input_binary_tensors: np.ndarray):
        """Load the dataset from binary tensors."""
        if self.binary_tensors is not None or self.precise_raw_data is not None:
            print("Warning: Overwriting existing dataset.")

        # check the shape of the input binary tensors
        correct_length = self.coordinate_precision * 2 + self.color_precision * 3
        if input_binary_tensors.shape[1] != correct_length:
            raise ValueError(
                (
                    "Input binary tensors have incorrect shape. Expected:"
                    f"{correct_length}, got:"
                    f"{input_binary_tensors.shape[1]}"
                )
            )

        self.binary_tensors = input_binary_tensors

        # convert the binary tensors to a pyarrow table
        x_binary = input_binary_tensors[:, : self.coordinate_precision]
        y_binary = input_binary_tensors[
            :, self.coordinate_precision : self.coordinate_precision * 2
        ]
        r_binary = input_binary_tensors[
            :,
            self.coordinate_precision * 2 : self.coordinate_precision * 2
            + self.color_precision,
        ]
        g_binary = input_binary_tensors[
            :,
            self.coordinate_precision * 2
            + self.color_precision : self.coordinate_precision * 2
            + self.color_precision * 2,
        ]
        b_binary = input_binary_tensors[
            :,
            self.coordinate_precision * 2
            + self.color_precision * 2 : self.coordinate_precision * 2
            + self.color_precision * 3,
        ]
        x = np.array(
            [
                int("".join(map(str, row)), 2) / pow(2, self.coordinate_precision - 1)
                for row in x_binary
            ]
        )
        y = np.array(
            [
                int("".join(map(str, row)), 2) / pow(2, self.coordinate_precision - 1)
                for row in y_binary
            ]
        )
        r = np.array([int("".join(map(str, row)), 2) for row in r_binary])
        g = np.array([int("".join(map(str, row)), 2) for row in g_binary])
        b = np.array([int("".join(map(str, row)), 2) for row in b_binary])

        self.precise_raw_data = pa.Table.from_arrays(
            [x, y, r, g, b], names=self.file_schema
        )

        return self
    
    def from_float_tensors(self, input_float_tensors: np.ndarray):
        """Load the dataset from float tensors."""
        if self.binary_tensors is not None or self.precise_raw_data is not None:
            print("Warning: Overwriting existing dataset.")

        # check the shape of the input binary tensors
        correct_length = 5
        if input_float_tensors.shape[1] != correct_length:
            raise ValueError(
                (
                    "Input float tensors have incorrect shape. Expected:"
                    f"{correct_length}, got:"
                    f"{input_float_tensors.shape[1]}"
                )
            )

        self.precise_raw_data = pa.Table.from_arrays(
            input_float_tensors.T, names=self.file_schema
        )

    def new_dataset(
        self,
        texture_file: str,
        spline_control_points: list,
        number_of_samples: int,
        sample_grid_width: int,
        sample_grid_height: int,
    ):
        """Generate a new dataset from a texture file and control points.
            Will normalize the positions to 0-1.

        Args:
            texture_file: str, the path to the texture image file.
            spline_control_points: list, the control points for the Catmull-Rom spline.
            number_of_samples: int, the number of samples to take from the spline.
            sample_grid_width: int, the width of the sample grid.
            sample_grid_height: int, the height of the sample grid.
        """
        # Generate the Catmull-Rom spline
        spline_sampled_coordinates = self._sample_catmull_rom_spline(
            spline_control_points, number_of_samples
        )

        # Load and resize the texture
        texture_image = Image.open(texture_file)
        texture_image_array = np.array(texture_image)

        # sample colors from the texture
        sampled_colors = self._map_grid_to_texture_colors(
            spline_sampled_coordinates,
            texture_image_array,
            sample_grid_width,
            sample_grid_height,
        )

        # finalizing the dataset
        combined_spline_dataset = np.hstack(
            (spline_sampled_coordinates, sampled_colors)
        )
        # scale the positions to 0-1
        combined_spline_dataset[:, :2] /= np.array(
            [sample_grid_width, sample_grid_height]
        )
        self.control_points_normalized = np.array(spline_control_points) / np.array(
            [sample_grid_width, sample_grid_height]
        )

        self.precise_raw_data = pa.Table.from_arrays(
            combined_spline_dataset.T, names=self.file_schema
        )

        return self

    def save_dataset(self, output_file: str = None):
        """Save the dataset to a CSV file."""
        if self.precise_raw_data is None:
            raise ValueError("No dataset loaded.")

        # save the dataset to a CSV file
        if output_file is None:
            output_file = self.input_file
        pc.write_csv(self.precise_raw_data, output_file)
        print(f"Dataset saved to {output_file}")

    def _sample_catmull_rom_spline(
        self, spline_control_points: list | np.ndarray, num_samples: int = 100
    ) -> np.ndarray:
        """Sample points from a Catmull-Rom spline defined by control points.

        Args:
            spline_control_points: list | np.ndarray, each element is a
                tuple of (x, y) coordinates of the control points.
            num_samples: int, number of points to sample from the spline.
        """

        # Extract x and y coordinates from control points
        spline_control_points = np.array(spline_control_points)
        x = spline_control_points[:, 0]
        y = spline_control_points[:, 1]

        # Generate parameter t
        t = np.linspace(0, 1, len(spline_control_points))

        # Create cubic spline for x and y coordinates
        cs_x = CubicSpline(t, x, bc_type="clamped")
        cs_y = CubicSpline(t, y, bc_type="clamped")

        # Generate uniformly spaced points
        t_fine = np.linspace(0, 1, num_samples)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)

        # Combine x and y coordinates
        sampled_spline_points = np.vstack((x_fine, y_fine)).T
        return sampled_spline_points

    def _map_grid_to_texture_colors(
        self,
        grid_sample_points: np.ndarray,
        texture_image_array: np.ndarray,
        sample_grid_width,
        sample_grid_height,
    ):
        """Map the grid sample points to the texture colors.

        Args:
            grid_sample_points: np.ndarray, shape (num_points, 2), each row
                is a (x, y) coordinate of the sample point.
            texture_array: np.ndarray, shape (height, width, 3), the texture image.
            grid_width: int, the width of the grid.
            grid_height: int, the height of the grid.
        """
        # obtain the height and width of the texture
        texture_height, texture_width, _ = np.shape(texture_image_array)
        print(texture_height, texture_width)
        # scale the spline points to the texture size
        scaled_spline_points = grid_sample_points * np.array(
            [texture_width / sample_grid_width, texture_height / sample_grid_height]
        )
        sampled_texture_colors = []
        for scaled_point in scaled_spline_points:
            x, y = scaled_point
            x = int(np.clip(x, 0, texture_width - 1))
            y = int(np.clip(y, 0, texture_height - 1))
            color = texture_image_array[y, x]  # Note: (y, x) indexing for image
            sampled_texture_colors.append(color)
        return np.array(sampled_texture_colors)

    def to_binary_tensors(self, masked: bool = False) -> np.ndarray:
        """Convert the dataset to binary tensors, with precision loss."""
        if self.precise_raw_data is None:
            raise ValueError("No dataset loaded.")
        self.binary_tensors = np.empty(
            (0, self.coordinate_precision * 2 + self.color_precision * 3), dtype=int
        )
        # load the dataset and convert to numpy array
        x = self.precise_raw_data["x"].to_numpy()
        y = self.precise_raw_data["y"].to_numpy()
        r = self.precise_raw_data["r"].to_numpy()
        g = self.precise_raw_data["g"].to_numpy()
        b = self.precise_raw_data["b"].to_numpy()
        # iterate all the rows in the input file
        for row_index in range(self.precise_raw_data.num_rows):
            # convert float to binary np array
            x_binary = np.binary_repr(
                int(x[row_index] * pow(2, self.coordinate_precision - 1)),
                width=self.coordinate_precision + 1,
            )[
                1:
            ]  # neglect the first sign bit
            y_binary = np.binary_repr(
                int(y[row_index] * pow(2, self.coordinate_precision - 1)),
                width=self.coordinate_precision + 1,
            )[1:]
            r_binary = np.binary_repr(
                int(r[row_index]), width=self.color_precision + 1
            )[
                1:
            ]  # neglect the first sign bit
            g_binary = np.binary_repr(
                int(g[row_index]), width=self.color_precision + 1
            )[1:]
            b_binary = np.binary_repr(
                int(b[row_index]), width=self.color_precision + 1
            )[1:]
            # concatenate the binary arrays
            binary_array = np.array(
                list(map(int, (x_binary + y_binary + r_binary + g_binary + b_binary)))
            )
            binary_array = binary_array > 0
            self.binary_tensors = np.vstack((self.binary_tensors, binary_array))

        if masked:
            if self.binary_mask is None:
                raise ValueError("No mask set.")
            self.binary_tensors[self.binary_mask] = 0

        # print("Binary conversion finished")

        return self.binary_tensors

    def to_float_tensors(self, masked: bool = False) -> np.ndarray:
        """Convert the binary tensors back to float tensors, with precision loss."""
        result = np.empty((0, 5))
        binary_tensors = self.to_binary_tensors(masked)
        for row_index in range(binary_tensors.shape[0]):
            row = binary_tensors[row_index]
            # print(row)
            x = int("".join(map(str, row[: self.coordinate_precision])), 2) / pow(
                2, self.coordinate_precision - 1
            )
            y = int(
                "".join(
                    map(
                        str,
                        row[self.coordinate_precision : self.coordinate_precision * 2],
                    )
                ),
                2,
            ) / pow(2, self.coordinate_precision - 1)
            r = int(
                "".join(
                    map(
                        str,
                        row[
                            self.coordinate_precision
                            * 2 : self.coordinate_precision
                            * 2
                            + self.color_precision
                        ],
                    )
                ),
                2,
            )
            g = int(
                "".join(
                    map(
                        str,
                        row[
                            self.coordinate_precision * 2
                            + self.color_precision : self.coordinate_precision * 2
                            + self.color_precision * 2
                        ],
                    )
                ),
                2,
            )
            b = int(
                "".join(
                    map(
                        str,
                        row[
                            self.coordinate_precision * 2
                            + self.color_precision * 2 : self.coordinate_precision * 2
                            + self.color_precision * 3
                        ],
                    )
                ),
                2,
            )
            result = np.vstack((result, np.array([x, y, r, g, b])))

        # print a sample of the result
        # print(result[:5])
        return result

    def to_raw_pa_table(self) -> pa.Table:
        """Output the dataset as a pandas DataFrame, no precision loss."""
        if self.precise_raw_data is None:
            raise ValueError("No dataset loaded.")
        return self.precise_raw_data

    def calculate_sparsity(self):
        """Calculate the sparsity of the dataset's binary array."""
        total_ones = np.sum(self.binary_tensors)
        total_elements = self.binary_tensors.size

        return total_ones / total_elements

    def plot_dataset(self, display: bool = True, save_as: str = None, figure: Figure = None, subplot_index: int = 111):
        """Plot the dataset."""
        if figure is None:
            figure = plt.figure()
        ax = figure.add_subplot(subplot_index)

        # Plot the sampled points
        x = self.precise_raw_data["x"].to_numpy()
        y = self.precise_raw_data["y"].to_numpy()
        color_r = self.precise_raw_data["r"].to_numpy() / 255
        color_g = self.precise_raw_data["g"].to_numpy() / 255
        color_b = self.precise_raw_data["b"].to_numpy() / 255
        ax.scatter(x, y, c=np.vstack((color_r, color_g, color_b)).T)

        # Plot the control points
        if self.control_points_normalized is None:
            print("No control points found. Skipping.")
        else:
            control_x = [point[0] for point in self.control_points_normalized]
            control_y = [point[1] for point in self.control_points_normalized]
            ax.plot(
                control_x,
                control_y,
                "ro--",
                label="Control Points",
            )

        if save_as is not None:
            plt.savefig(save_as)

        if display:
            plt.show()
            

    def _set_coordinates_mask(self, mask_ratio: float = 1.0):
        """Mask the coordinates of the dataset."""

        # mask the coordinates
        # True means masked
        mask = np.random.choice(
            [False, True],
            size=(self.binary_tensors.shape[0], self.coordinate_precision * 2),
            p=[1 - mask_ratio, mask_ratio],
        )
        # extend the mask to match the dataset shape
        # the extended part should be False
        mask_extend = np.zeros(
            (self.binary_tensors.shape[0], self.color_precision * 3), dtype=bool
        )
        # concatenate the mask
        self.binary_mask = np.concatenate((mask, mask_extend), axis=1)

    def _set_colors_mask(self, mask_ratio: float = 1.0):
        """Mask the colors of the dataset."""

        # mask the colors
        # True means masked
        mask = np.random.choice(
            [False, True],
            size=(self.binary_tensors.shape[0], self.color_precision * 3),
            p=[1 - mask_ratio, mask_ratio],
        )
        # extend the mask to match the dataset shape
        # the extended part should be False
        mask_extend = np.zeros(
            (self.binary_tensors.shape[0], self.coordinate_precision * 2), dtype=bool
        )
        # concatenate the mask
        self.binary_mask = np.concatenate((mask_extend, mask), axis=1)

    def _set_full_mask(self, mask_ratio: float = 1.0):
        """Mask the full dataset."""

        # mask the full dataset
        # True means masked
        self.binary_mask = np.random.choice(
            [False, True],
            size=(
                self.binary_tensors.shape[0],
                self.coordinate_precision * 2 + self.color_precision * 3,
            ),
            p=[1 - mask_ratio, mask_ratio],
        )

    def set_mask(self, mask_type: str, mask_ratio: float = 1.0):
        """Set the mask for the dataset."""
        # check the mask type
        if mask_type not in self.mask_types:
            raise ValueError(f"Mask type {mask_type} is not supported.")

        # check binary tensors
        if self.binary_tensors is None:
            self.to_binary_tensors()

        match mask_type:
            case "coordinates":
                self._set_coordinates_mask(mask_ratio)
            case "colors":
                self._set_colors_mask(mask_ratio)
            case "full":
                self._set_full_mask(mask_ratio)
            case "none":
                self.binary_mask = np.zeros(self.binary_tensors.shape, dtype=bool)


if __name__ == "__main__":

    # Define the 2D grid limits
    GRID_WIDTH = 6
    GRID_HEIGHT = 6
    NUM_POINTS = 100

    # Define control points for the Catmull-Rom spline
    control_points = [(1, 1), (2, 3), (4, 4), (5, 1), (6, 3)]

    # Generate the dataset
    example = MinimalBTSPDataset(32, 8)
    example.new_dataset(
        "./image.png",
        control_points,
        NUM_POINTS,
        GRID_WIDTH,
        GRID_HEIGHT,
    )

    dataset = example.precise_raw_data

    # print("Sampled dataset:")
    print(dataset[:5])  # Print the first 5 rows of the dataset

    example.save_dataset("dataset.csv")
    example.from_file("dataset.csv")
    example.plot_dataset()

    # test binary conversion
    for mask in example.mask_types:
        example.set_mask(mask, 1)
        tensors = example.to_float_tensors(masked=True)
        example_from_tensors = MinimalBTSPDataset(32, 8)
        example_from_tensors.from_float_tensors(tensors)
        example_from_tensors.plot_dataset()
