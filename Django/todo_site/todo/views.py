

from django.contrib import messages
from django.shortcuts import render, redirect

from .forms import TodoForm
from .models import Todo

def index(request):
    item_list = Todo.objects.order_by("-date")
    if request.method == "POST":
        form = TodoForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, ("Item has been added to list!"))
            return render(request, 'todo/index.html', {'item_list': item_list})
        
    form = TodoForm()
    page={
        "forms": form,
        "list": item_list,
        "title": "TODO LIST",
    }
    return render(request, 'todo/index.html', page)

def remove(request, item_id):
    item = Todo.objects.get(id=item_id)
    item.delete()
    messages.info(request, ("Item has been deleted!"))
    return redirect('todo')