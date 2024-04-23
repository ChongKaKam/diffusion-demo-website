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

#### 3 Flask: run

Run the command after vue preparation is finished.

```shell
cd flask_backend
python3 app.py
```

 
