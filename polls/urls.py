from django.urls import path

from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('home/', views.home, name='home'),
    path('library/', views.library, name='library'),
    path('library/<int:book_index>/', views.library_detail, name='library-detail'),
    path('library-calculator/', views.library_calculator, name='library-calculator'),
    path('library-calculator/<int:pk>/', views.library_calculator_detail, name='library-calculator-detail'),
    path('create/', views.create_task, name='create-task'),
    path('api/start/<int:pk>/', views.api_start, name='api-start'),
    path('api/stop/<int:pk>/', views.api_stop, name='api-stop'),
    path('api/progress/<int:pk>/', views.api_progress, name='api-progress'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
]
