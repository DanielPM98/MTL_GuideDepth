import torch


# Decorator to transform a list argument to tensor
def list_to_tensor(func):
    def wrapper(lst):
        # Convert the list to a tensor
        tensor = torch.Tensor(lst)
        # Call the original function with the tensor
        return func(tensor)
    # Return the modified function
    return wrapper
