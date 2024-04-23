<template>
    <div class="col-frame">
      <p class="img-title">Face Mask</p>
      <div v-loading="canvaseLoading">
        <canvas ref="canvas"
              @mousedown="startDrawing" 
              @mouseup="stopDrawing" 
              @mousemove="draw"
              width="256"
              height="256"></canvas>
      </div>
      <el-button class="canvas-button" size="small" type="info" plain @click="setImg">get mask</el-button>
      <el-button class="canvas-button" size="small" type="warning" plain @click="generateImage" >generate</el-button>
      <div class="brush-tool-box">
        <div class="brush-slider">
          <p>Brush Size</p>
          <el-slider v-model="brushWidth" size="small" :step="1" :min="6" :max="16" show-stops/>
        </div>
        <div style="margin-top: 4px">
          <el-radio-group v-model="brushColor" fill="#73767a" >
            <el-radio-button label="Face" value="orange"/>
            <el-radio-button label="Hair" value="brown"/>
            <el-radio-button label="Lip" value="red" />
            <el-radio-button label="Ear" value="green" />
          </el-radio-group>
        </div>
      </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const canvaseLoading = ref(false)
const drawing = ref(false);
const canvas = ref(null);
const context = ref(null);
const lastX = ref(0);
const lastY = ref(0);
const brushColor = ref('black');
const brushWidth = ref(6);
const backgroundImage = ref(null);

onMounted(() => {
  context.value = canvas.value.getContext('2d');
  context.value.lineWidth = brushWidth.value;
  context.value.lineCap = 'round';
  context.value.strokeStyle = brushColor.value;
});

const startDrawing = (e) => {
  drawing.value = true;
  [lastX.value, lastY.value] = [e.offsetX, e.offsetY];
};

const stopDrawing = () => {
  drawing.value = false;
};

const draw = (e) => {
  if (!drawing.value) return;
  context.value.lineWidth = brushWidth.value;
  context.value.strokeStyle = brushColor.value;
  context.value.beginPath();
  context.value.moveTo(lastX.value, lastY.value);
  context.value.lineTo(e.offsetX, e.offsetY);
  context.value.stroke();
  [lastX.value, lastY.value] = [e.offsetX, e.offsetY];
};
// Buttom handle function
const setImg = async () => {
  try {
    canvaseLoading.value = true;
    // setTimeout(async()=>{canvaseLoading.value=false}, 3000)
    const postResponse = await axios.post('/gen-image', {type:'mask', image: 'none'});
    if (postResponse.status===200){
      const getResponse = await axios.get('/mask-url');
      const img = new Image();
      canvaseLoading.value = false;
      img.src = getResponse.data.url;
      img.onload = () => {
        context.value.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);
      };
    } else{
      canvaseLoading.value = false;
      console.error('POST request failed');
    }
  } catch (error) {
    console.error(error);
  }
};
const generateImage = async() => {
  try{
    const canvasImg = canvas.value.toDataURL('image/mask.png');
    const response = await axios.post('/gen-image', {type:'final', image: canvasImg});
  }catch(error){
    console.error(error);
  }
};
</script>
  
<style scoped>
  canvas {
    border: 2px solid black;
    background-color: white;
    width: 260px;
    height: 260px;
    vertical-align: top;
  }
  .col-frame{
    width: 260px;
    align-items: center;
    text-align: center;
    vertical-align: top;
  }
  .img-title{
    font-size: 32px;
    text-align: center;
    margin: 12px;
  }
  .canvas-button{
    margin-top: 24px;
    width: 100px;
    height: 40px;
    font-size: 20px;
    color: rgb(58, 58, 58);
    margin-left: 12px;
    margin-right: 12px;
  }
  .brush-title{
    font-size: 24px;
    margin: 8px;
  }
  .brush-tool-box{
    margin-top: 12px;
    padding-bottom: 8px;
    border-radius: 16px;
    border: 2px #cecece dashed;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .brush-button{
    margin-top: 8px;
    margin-bottom: 8px;
    width: 100px;
    height: 40px;
    font-size: 20px;
    color: rgb(58, 58, 58);
    margin-left: 12px;
    margin-right: 12px;
  }
  .brush-slider{
    margin-top: 8px;
    width: 200px;
    font-size: 18px;
  }
</style>