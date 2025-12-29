from flask import Blueprint, request, render_template
from research.logic import init_environment, load_llm, load_retriever, query_with_context

bp = Blueprint('main', __name__)

init_environment()
llm = load_llm()
retriever = load_retriever()

@bp.route('/', methods=['GET', 'POST'])
def index():
    user_query = ""
    answer = ""
    if request.method == 'POST':
        user_query = request.form['query']
        result = query_with_context(user_query, retriever, llm)
        answer = result.split("Assistant:")[-1].strip() if "Assistant:" in result else result
    return render_template("index.html", query=user_query, answer=answer)
