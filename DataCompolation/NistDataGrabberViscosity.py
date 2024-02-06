import numpy as np
import ThermoMlReader as tml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def OrrickEarbarTransform(temperature,Viscoisty):
    y=np.log(Viscoisty)
    temperature=temperature.reshape(-1,1)
    X=np.hstack((temperature**-1,np.ones_like(temperature)))
    return X,y

def FitOrrickEarbar(X,y):

    model=LinearRegression(fit_intercept=False)

    model.fit(X,y)
    #print fit 
    r2=r2_score(y,model.predict(X))
    B,A=model.coef_
    return A,B

class TargetConstructor():
    def __init__(self,StateEquations):
        self.StateEquations=StateEquations
    def GenerateTargets(self,FilePath,step=1):

    
        tml_parser = tml.ThermoMLParser(FilePath)

        tml_parser.extract_properties()
        tml_parser.extract_equation_details()
        Properties=tml_parser.get_properties()
        

        idx = np.where(np.isin(Properties['equation_names'], list(self.StateEquations.keys())))[0]
        EqName = [Properties['equation_names'][i] for i in idx]

        idx=Properties['equation_names'].index(EqName[0])

        Params=Properties['equation_parameters'][idx]
        VarRange=Properties['variable_ranges'][idx]


        # Generate a linspace for each variable range and stack them horizontally.
        x = np.column_stack([np.arange(var_min, var_max, step) for var_min, var_max in VarRange])

        StateEquation=self.StateEquations[EqName[0]]
        Y=StateEquation.run(VarRange,Params,step=step)

        #compute A and B
        Xf,yf=OrrickEarbarTransform(x,Y)
        A,B=FitOrrickEarbar(Xf,yf)

        self.X=x
        self.y=Y
        self.A=A
        self.B=B
        return A,B,x,Y


