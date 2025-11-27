<h1> Heart Disease Detection </h1>
In this project we aim to predict the presence of a heart disease based on the the deterministic features : age, sex, cholesterol level, fasting blood sugar, resting ECG, Max HR, ST depression and the qualitative features : chest pain type, exercise angina. Here the chest pain type is categorized in to 4 levels - 1 to 4, where each level indicates a specific type of chest pain encountered by the patient. The feature exercise angina signifies the presence of pain during exercising, taking the values 0 and 1.

<h2>Requirements</h2>
See the pyproject.toml file

<h2>Build</h2>
To build the project you need to have the build package installed. To get the build package run <code>pip install build</code>. With the build package you can then run <code> python -m build </code> from the top directory to aquire the necessary dependencies. This will generate a subfoler called <code>dist</code> in which a wheel file with the file extension <code>.whl</code> will be located. To unpack the wheel file and install the dependencies run <code> pip install dist/example.whl </code>. After this you will be able to run the main file, i.e. <code>python regression_code.py </code>.

<h2>Visualisations</h2>
The visualisation like the feature importance, ROC/AUC curve are produced and can be viewed in the <code>/results</code> directory