from flask import Flask, render_template

# Flask uygulamasının bildirimi
app = Flask(__name__)

@app.route('/main', methods=['GET', 'POST'])
def main():
    return render_template('main.html')

@app.route('/DataSet', methods=['GET', 'POST'])
def DataSet():
    return render_template('DataSet.html')

@app.route('/DataSetVisualization', methods=['GET', 'POST'])
def DataSetVisualization():
    return render_template('DataSetVisualization.html')

@app.route('/TestTrain', methods=['GET', 'POST'])
def TestTrain():
    return render_template('TestTrain.html')

@app.route('/KNearestNeighbours', methods=['GET', 'POST'])
def KNearestNeighbours():
    return render_template('KNearestNeighbours.html')

@app.route('/NaiveBayesClassifier', methods=['GET', 'POST'])
def NaiveBayesClassifier():
    return render_template('NaiveBayesClassifier.html')

@app.route('/SupportVectorMachines', methods=['GET', 'POST'])
def SupportVectorMachines():
    return render_template('SupportVectorMachines.html')

@app.route('/LogisticRegression', methods=['GET', 'POST'])
def LogisticRegression():
    return render_template('LogisticRegression.html')

@app.route('/RandomForest', methods=['GET', 'POST'])
def RandomForest():
    return render_template('RandomForest.html')

@app.route('/DecisionTrees', methods=['GET', 'POST'])
def DecisionTrees():
    return render_template('DecisionTrees.html')

@app.route('/DecisionTreeRegression', methods=['GET', 'POST'])
def DecisionTreeRegression():
    return render_template('DecisionTreeRegression.html')

@app.route('/SupportVectorRegression', methods=['GET', 'POST'])
def SupportVectorRegression():
    return render_template('SupportVectorRegression.html')

@app.route('/LinearRegression', methods=['GET', 'POST'])
def LinearRegression():
    return render_template('LinearRegression.html')

"""@app.route('/', methods=['POST'])
def predict():"""

# Uygulamayı çalıştırmaya yaramaktadır
if __name__ == '__main__':
    app.debug = True
    app.run()