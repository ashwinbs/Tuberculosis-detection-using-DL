from django.shortcuts import render
from tuberculosisApp.models import User
from django.db.models import Q, Sum
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from tuberculosisApp import predict
import shutil
import os

# Create your views here.
def index(request):
    return render(request,'index.html')

def user(request):
    return render(request,'user/index.html')


def registration(request):
    return render(request,'user/registration.html')


def saveUser(request):
    if request.method == 'POST':
        farmername = request.POST['uname']
        contactNo = request.POST['contactNo']
        emailId = request.POST['emailId']
        address = request.POST['address']
        username = request.POST['username']
        password = request.POST['password']

        user = User.objects.filter(
            Q(email=emailId) | Q(contact=contactNo) | Q(user_name=username)
        ).first()

        has_error = False
        error = ''

        if user != None and user.user_name == username:
            has_error = True
            error = 'Duplicate user name'

        if user != None and user.email == emailId:
            has_error = True
            error = 'Duplicate email'

        if user != None and user.contact == contactNo:
            has_error = True
            error = 'Duplicate contact number'

        if has_error:
            return render(request, "user/registration.html", {'error': error})

        user = User(name=farmername, contact=contactNo, email=emailId,
                    address=address, user_name=username, password=password)
        user.save()

        return render(request, "user/registration.html", {'success': 'User Added Successfully'})
    else:
        return render(request, 'user/registration.html')


def userlogin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = User.objects.values_list('password', 'id', 'name').\
            filter(user_name=request.POST['username'])

        user = User.objects.filter(
            user_name=username, password=password).first()

        if user == None:
            return render(request, 'user/index.html', {'error': 'Invalid login credentials'})

        request.session['userid'] = user.id
        request.session['userName'] = user.name

        return render(request, 'user/userHome.html')

    else:
        return render(request, 'user/index.html')


def uploadImage(request):
    return render(request,'user/upload.html')

def homepage(request):
    return render(request,'user/userHome.html')


def home(request):

    if request.method == "GET":
        return render(request, 'home.html')

    if request.method == "POST":
        image = request.FILES['test1']

        shutil.rmtree(os.getcwd() + '\\media')

        path = default_storage.save(
            os.getcwd() + '\\media\\input.png', ContentFile(image.read()))
        
        print(path)
        result = predict.process()
        

    return render(request, "user/result.html", {'result': result,'path':path})


def testagain(request):
    return render(request, "user/upload.html")