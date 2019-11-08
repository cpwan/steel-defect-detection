from flask import Flask, render_template, request
import glob
app = Flask(__name__,static_folder='../data')

imagepath=glob.glob('../data/mask_colored/test/*.png')
i=0

@app.route('/', methods=['GET','POST'])
def render():
    global i
    global imagepath
    if request.method == 'POST':
        if request.form['submit_btn'] == 'previous':
            i-=1
        elif request.form['submit_btn'] == 'next':
            i+=1
    i%=len(imagepath)
    filename= imagepath[i].split('/')[-1]
    
    return render_template('./imshow.html', path_to_img=filename.split('.')[0])

@app.route('/<ids>', methods=['GET','POST'])
def render_by(ids):
    global i
    global imagepath
    if request.method == 'POST':
        if request.form['submit_btn'] == 'previous':
            i-=1
        elif request.form['submit_btn'] == 'next':
            i+=1
    i%=len(imagepath)
    filename= imagepath[i].split('/')[-1]
    
    return render_template('./imshow.html', path_to_img=ids)




if __name__ == '__main__':

    app.run(debug = True )
