import pyautogui as pg
import numpy as np
import time
import json

#load the data from json file
with open('HydrocarbonDataProcessed.json', 'r') as infile:
    HydrocarbonData = json.load(infile)



try:
    indexDone = np.load('indexDone.npy')
    print(indexDone)
    #delete all indexes before indexDone
    HydrocarbonData = HydrocarbonData[indexDone:]
except:
    indexDone = 0

indexsToRemove = []

pg.sleep(3)



for i,dict in enumerate(HydrocarbonData):
    np.save('indexDone.npy',i + indexDone)
    #remove ' and " from name
    
    name=dict['MolName']
    name=name.replace("'","")
    name=name.replace('"','')

    print(name)

    if name == '':
        indexsToRemove.append(i)
        continue

    print(len(HydrocarbonData)- i)
    #press control n
    pg.hotkey('ctrl','n')
    pg.sleep(0.1)
    #ctrl a wait
    pg.hotkey('ctrl','a')
    pg.sleep(.5)
    #type name
    pg.typewrite(name)
    pg.sleep(0.1)
    #press enter
    pg.press('enter')
    pg.sleep(0.5)
    try:
        if pg.locateOnScreen('Scraper\DrawData.png',confidence=0.8)!=None:
            #press escape
            pg.press('esc')
            pg.sleep(0.5)
            pg.press('esc')
            pg.sleep(0.5)
            
            indexsToRemove.append(i)
            continue
    except:
        pass
    try:
        if pg.locateOnScreen('Scraper\ErrorBox.png',confidence=0.8)!=None:
            #press escape
            pg.press('esc')
            pg.sleep(0.1)
            pg.press('esc')

            indexsToRemove.append(i)
            continue
    except:
        pass
    #press control g
    pg.hotkey('ctrl','g')
    pg.sleep(0.1)

    Eval=False
    while Eval==False:
        try:
            try:
                if pg.locateOnScreen('Scraper\EvaluatedData.png',confidence=0.6)!=None:
                    Eval=True
                    print('Evaluated')

            except:
                pass
                #if pg.locateOnScreen('Scraper\EvaluatedData2.png',confidence=0.6)!=None:
                #    Eval=True
                #    print('Evaluated')
            pg.sleep(0.1)
        except:
            pass

    pg.sleep(1)

    Eval=False

    while Eval==False:
        pg.hotkey('ctrl','s',pauses=0.1)
        pg.sleep(.5)
        try:
            if pg.locateOnScreen('Scraper\EvalReact.png',confidence=0.8)!=None:
                #press escape
                pg.press('right')
                pg.sleep(0.1)
                pg.press('enter')
                pg.sleep(0.1)
        except:
            pass
        try:
            if pg.locateOnScreen('Scraper\SaveBoxOpen.png',confidence=0.7)!=None:
                Eval=True
        except:
            pass
    Eval=False

    #type name
    print(name)

    pg.typewrite(name,interval=.05)
    pg.sleep(0.5)
    #enter
    pg.press('enter')
    pg.sleep(1)

