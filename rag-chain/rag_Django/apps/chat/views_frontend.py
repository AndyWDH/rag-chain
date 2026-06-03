from django.shortcuts import render


def index(request):
    """渲染前端对话界面"""
    return render(request, 'index.html')