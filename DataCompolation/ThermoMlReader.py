import xmltodict
import numpy as np

class ThermoMLParser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.doc = None
        self.property_groups = []
        self.property_names = []
        self.property_phase = []
        self.property_equation_names = []
        self.property_Eequation_names = []
        self.VarSymbolsList = []
        self.VarRangeList = []
        self.ParamsList = []
        
        self.load_xml()

    def load_xml(self):
        with open(self.file_name) as fd:
            self.doc = xmltodict.parse(fd.read())

    def step_in(self, dictionary):
        if len(list(dictionary.keys())) != 1:
            raise ValueError('The dictionary has more than one key!')
        return list(dictionary.keys())[0]

    def extract_properties(self):
        propertylist = self.doc['DataReport']['PureOrMixtureData']
        for property in propertylist:
            propnames = property['Property']['Property-MethodID']['PropertyGroup']
            self.property_groups.append(self.step_in(propnames))
            self.property_names.append(propnames[self.step_in(propnames)]['ePropName'])
            self.property_phase.append(property['Property']['PropPhaseID']['ePropPhase'])
            self.extract_equation_names(property)

    def extract_equation_names(self, property):
        if 'Equation' in property:
            equations = property['Equation']
            equations = equations if isinstance(equations, list) else [equations]
            for equation in equations:
                self.property_equation_names.append(equation.get('sEqName', ''))
                self.property_Eequation_names.append(equation.get('eEqName', ''))

    def extract_equation_details(self):
        for property in self.doc['DataReport']['PureOrMixtureData']:
            equations = property.get('Equation', [])
            if not isinstance(equations, list):  # Ensuring equations is a list
                equations = [equations]

            for equation in equations:
                self.extract_variables_and_params(equation)

    def extract_variables_and_params(self, equation):
        variable_symbols, variable_ranges, equation_parameters = [], [], []

        eq_variables = equation.get('EqVariable', [])
        eq_variables = eq_variables if isinstance(eq_variables, list) else [eq_variables]
        for var in eq_variables:
            symbol = var.get('sEqSymbol')
            min_val = float(var.get('nEqVarRangeMin')) if var.get('nEqVarRangeMin') else None
            max_val = float(var.get('nEqVarRangeMax')) if var.get('nEqVarRangeMax') else None
            variable_symbols.append(symbol)
            variable_ranges.append((min_val, max_val))

        eq_parameters = equation.get('EqParameter', [])
        eq_parameters = eq_parameters if isinstance(eq_parameters, list) else [eq_parameters]
        for param in eq_parameters:
            value = float(param.get('nEqParValue')) if param.get('nEqParValue') else None
            if value is not None:
                equation_parameters.append(value)

        self.VarSymbolsList.append(variable_symbols)
        self.VarRangeList.append(variable_ranges)
        self.ParamsList.append(equation_parameters)

    def save_to_file(self):
        data = list(zip(self.property_groups, self.property_names, self.property_phase, self.property_equation_names, self.property_Eequation_names))
        np.savetxt('PropertyList.txt', data, delimiter='\t', fmt='%s')

    def get_properties(self):
        return {
            'property_names': self.property_names,
            'property_phase': self.property_phase,
            'variable_symbols': self.VarSymbolsList,
            'variable_ranges': self.VarRangeList,
            'equation_parameters': self.ParamsList,
            'equation_names': self.property_equation_names
        }



