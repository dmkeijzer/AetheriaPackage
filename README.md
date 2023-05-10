# Project11_VTOL

This is the repostiory for the DSE of 2023 where a long range VTOL will be designed keeping in mind aircrashworthiness.

# Requirements for code repository

To able to execute the code, a requirements file has been uploaded called `requirements.txt`. To be able to install all the packages use the following command in your anaconda prompt. 

``` conda create --name <new_environment_name> --file requirements.txt```


Then use this environment when running your code. Please consult documentation on your IDE on how to do this.

# Adding new functions

When you add a new function to the codebase, please include a docstring using the Google format, as shown in the example below:

```python
def find_mac(S, b, taper):
    """ Computes the mean aerodynamic chord

    :param S: Wing surface area [m^2]
    :type S: float
    :param b: Wingspan [m]
    :type b: float
    :param taper: Wing taper
    :type taper: float
    :return: Mean Aerodynamic chord
    :rtype: float
    """    
    cavg = S / b
    cr = 2 / (1 + taper) * cavg
    mac = 2/3 * cr * (1 + taper + taper ** 2) / (1 + taper)
    return mac

```
When doing this we can automatically create documentation using the `spinx` library

## Note

If you're manually installing the packages, note that the `import pdffit` has to be installed as `pip install pyprobabilitydistributionfit`
