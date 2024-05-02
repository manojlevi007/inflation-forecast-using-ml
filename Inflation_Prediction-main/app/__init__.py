import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import date

#Create the flask app
app = Flask(__name__, template_folder= 'templates', static_url_path = "/images", static_folder = "images")


@app.route("/",  methods= ['GET', 'POST'])
def Hello(month= 0, model= 1):
    if request.method== 'GET':
        return render_template("home.html")
    else:
        data= request.form
        if data['month'] != '':
            month= int(data['month']) 
        if data['model'] != '':
            model= int(data['model'])
        if month> 12:
            return render_template("home.html", valid= 1)
        elif month<= 0:
            return render_template("home.html", valid= 2)
        else:
            plotGenerator(model, month)
            return render_template("home.html", valid= 3, model= model)



def plotGenerator(model, month):
    latestdata = pd.read_csv("data/last20data.csv")['Data'].tolist()
    prediction= pd.read_csv('../app/data/model{}.csv'.format(model))['Data'].tolist()[:month]
    prediction.insert(0, latestdata[-1])
    import datetime
    s = '04/2019'

    x1= [datetime.datetime.strptime(s, '%m/%Y') +relativedelta(months=i) for i in range(20)]
    x2= [datetime.datetime.strptime(s, '%m/%Y') + relativedelta(months= 19+ i) for i in range(month+ 1)]
    plt.plot(x1, latestdata, 'r', x2, prediction, 'b--')
    plt.xticks(rotation= 30)
    plt.grid()
    plt.title('Inflation prediction over {} months'.format(month))
    plt.xlabel('Date')
    plt.ylabel('Inflation Rate in %')
    plt.savefig('images/prediction{}.png'.format(model))
    plt.clf()



if __name__ == '__main__':
   app.run(debug= True)
