"""
Module: Input Preprocessing Utility
Author: Silvio Christian, Joe
Description:
    Helper function to restructure Streamlit inputs into the 
    array format expected by the Scikit-Learn model.
"""

def preprocess_input(preg, glu, bp, skin, ins, bmi, dpf, age):
    """
    Organizes individual health metrics into a feature vector.
    
    Args:
        preg, glu, bp, skin, ins, bmi, dpf, age: Patient health metrics.
        
    Returns:
        list: Ordered list of features corresponding to the model's training columns.
    """
    return [preg, glu, bp, skin, ins, bmi, dpf, age]
