import math

def round_up_to_5_sig_figs(number):
    if number == 0:
        return 0
    else:
        # Get the exponent of the number
        exponent = math.floor(math.log10(abs(number))) 
        # Calculate the multiplier needed to get the number to 5 significant figures
        multiplier = 10 ** (4 - exponent) 
        # Round up the number to the nearest multiple of the multiplier
        rounded_number = math.ceil(number * multiplier) / multiplier 
        return rounded_number

def round_down_to_5_sig_figs(number):
    if number == 0:
        return 0
    else:
        # Get the exponent of the number
        exponent = math.floor(math.log10(abs(number))) 
        # Calculate the multiplier needed to get the number to 5 significant figures
        multiplier = 10 ** (4 - exponent) 
        # Round down the number to the nearest multiple of the multiplier
        rounded_number = math.floor(number * multiplier) / multiplier 
        return rounded_number
    