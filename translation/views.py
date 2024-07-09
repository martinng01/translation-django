import threading
from django.http import JsonResponse
from django.shortcuts import render
from .logic import langchain_exists, translate, database_exists, setup_backend

# https://python.plainenglish.io/django-langchained-e53aab3ad6bf


def index(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        result = translate(query)
        return JsonResponse({'translation': result})
    return render(request, 'translation/index.html')


def db_status(request):
    ready = database_exists() and langchain_exists()
    status = {
        'exists': ready,
        'message': 'Database exists' if ready else 'Database is being built'
    }
    return JsonResponse(status)


def build_db(request):
    thread = threading.Thread(target=setup_backend)
    thread.start()
    return JsonResponse({'status': 'Building database...'})
