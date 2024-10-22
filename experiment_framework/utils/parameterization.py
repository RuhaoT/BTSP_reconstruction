"""Aims to provide efficient way to control and generate experiment paramters.

"""

import itertools


def iterate_dict(dictionary: dict):
    """This function iterates over combinations of dictionary values.

    Given a dictionary with elements as lists, this function generates
    single-element dictionary with all possible combinations of elements.

    Args:
        dictionary (dict): The input dictionary.
    """
    dict_keys = dictionary.keys()
    dict_values = dictionary.values()

    # convert single values to list
    dict_values = [
        [value] if not isinstance(value, list) else value for value in dict_values
    ]

    value_combinations = list(itertools.product(*dict_values))

    result = [
        dict(zip(dict_keys, value_combination))
        for value_combination in value_combinations
    ]
    return result


def recursive_iterate_dict(dictionary: dict):
    """This function iterates over combinations of dictionary values.

    Given a dictionary with elements as lists or dictionaries, this function
    generates single-element dictionary with all possible combinations of
    elements in the dictionary or its sub-dictionaries recursively.

    Args:
        dictionary (dict): The input dictionary.
    """

    def recursive_dict_combinations(subdict: dict):
        """This is the helper function of recursive_iterate_dict.

        This function recursively calls itself to iterate over all subsidiary
        dictionaries of the input dictionary and return all value combinations
        in the current subdict.
        """
        keys = []
        values = []

        for key, value in subdict.items():
            if isinstance(value, list):
                keys.append(key)
                values.append(value)
            elif isinstance(value, dict):
                sub_combinations = recursive_dict_combinations(value)
                keys.append(key)
                values.append(sub_combinations)
            else:
                # single value situation
                keys.append(key)
                values.append([value])

        combinations = itertools.product(*values)

        return [dict(zip(keys, combination)) for combination in combinations]

    return recursive_dict_combinations(dictionary)


# provide a module test
if __name__ == "__main__":
    test_dict_with_subdict = {
        "a": [1, 2],
        "b": {"c": [3, 4], "d": [5, 6]},
    }
    test_dict = {
        "a": [1, 2],
        "b": [3, 4],
    }
    print(recursive_iterate_dict(test_dict_with_subdict))
    print(iterate_dict(test_dict))
