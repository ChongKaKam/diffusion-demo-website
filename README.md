# diffusion-demo-website
A toy demo of diffusion model

### Quick start

#### 1 Preparation:

Before you get statred, it is necessary to check the environment:

+ Flask ~= 3.0
+ Vue = 3.4.23
+ ...

#### 2 Vue: Generate Dist 

Run the command below to generate `dist`:

```shell
cd vue_frontend
npm run build
# the structure tree:
dist/
├── _static/
│   ├── icons/
│   ├── xxx.css
│   └── xxx.js
├── favicon.ico
└── index.html
```

#### 3 Flask: prepare and run

After you get the `dist`, you should copy it into `flask_backend/template`, then the flask app can find the right files.

Run the command after preparation is finished.

```shell
cd flask_backend
python3 app.py
```

 
