from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired

from pipelines import pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this-is-test'
Bootstrap(app)

TASK_NAME = "e2e-qg-v2"

nlp = pipeline('question-generation', 'runs/t5-small--hl-plus-rules', 't5_qg_tokenizer')

class InputForm(FlaskForm):
    text = TextAreaField('Enter your text:', validators=[DataRequired()], render_kw={"rows": 8, "cols": 11})
    submit = SubmitField('Generate')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    questions = ""

    if form.validate_on_submit():
        text = form.text.data
        questions = nlp(text)
        # redirect the browser to another route and template
        if TASK_NAME == 'qg':
            return render_template('home.html', form=form, questions=questions)
        elif TASK_NAME == 'e2e-qg':
            return render_template('home2.html', form=form, questions=questions)
        else:
            raise Exception("Please specify task name...")

    return render_template('home.html', form=form, questions=questions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # python -m flask run -h '10.2.33.108'
