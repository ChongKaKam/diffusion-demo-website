<template>
    <div class="col-frame">
      <p class="img-title">Face Mask</p>
      <div v-loading="canvaseLoading" style="position: relative;">
        <el-image style="width: 260px; height: 260px; background-color: white;" :src="url" fit="contain">
          <template #error >
            <el-image src="_static/icons/photo.png" class='error-img'></el-image>
          </template>
        </el-image>
        <canvas ref="canvas"
              @mousedown="startDrawing" 
              @mouseup="stopDrawing" 
              @mousemove="draw"
              width="256"
              height="256"
              style="background-color: transparent; position:absolute; top: 0; left: 0;"></canvas>
      </div>
      <el-button class="canvas-button" size="small" type="info" plain @click="setImg">get mask</el-button>
      <el-button class="canvas-button" size="small" type="warning" plain @click="generateImage" >generate</el-button>
      <div class="brush-tool-box">
        <div class="brush-slider">
          <p>Brush Size</p>
          <el-slider v-model="brushWidth" size="small" :step="1" :min="6" :max="16" show-stops/>
        </div>
        <div class="selectBar">
          <el-radio-group v-model="brushColor" fill="#73767a"  >
            <el-radio-button label="Skin" value="#cc0000"/>
            <el-radio-button label="Hair" value="#0000cc"/>
            <el-radio-button label="Nose" value="#4c9900" />
            <el-radio-button label="Cloth" value="#00cc00" />
          </el-radio-group>
        </div>
        <div class="selectBar">
          <el-radio-group v-model="brushColor" fill="#73767a" >
            <el-radio-button label="eye_L" value="#3333ff"/>
            <el-radio-button label="eye_R" value="#cc00cc" />
            <el-radio-button label="mouth" value="#66cc00" />
            <!-- <el-radio-button label="browL" value="#ff3399" /> -->
            <!-- <el-radio-button label="Blcakground" value="#000000"/> -->
          </el-radio-group>
        </div>
        <div class="selectBar">
          <el-radio-group v-model="brushColor" fill="#73767a" >
            <el-radio-button label="Blcakground" value="#000000"/>
            <el-radio-button label="Eraser" value="eraser"/>
          </el-radio-group>
        </div>
      </div>
    </div>
</template>

<script setup>
import { ref, onMounted , onBeforeUnmount} from 'vue';
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
const url = ref('')
const props = defineProps(['message'])
const emitMessage = defineEmits(['changeMessage']);

onMounted(() => {
  context.value = canvas.value.getContext('2d');
  context.value.lineWidth = brushWidth.value;
  context.value.lineCap = 'round';
  context.value.strokeStyle = brushColor.value;
});

onBeforeUnmount(()=>{
  clearCanvas()
})

const startDrawing = (e) => {
  drawing.value = true;
  [lastX.value, lastY.value] = [e.offsetX, e.offsetY];
};

const stopDrawing = () => {
  drawing.value = false;
};

const draw = (e) => {
  if (!drawing.value) return;
  if (brushColor.value ==='eraser'){
    context.value.globalCompositeOperation = 'destination-out';
    context.value.lineWidth = brushWidth.value;
  }else{
    context.value.globalCompositeOperation = 'source-over';
    context.value.lineWidth = brushWidth.value;
    context.value.strokeStyle = brushColor.value;
  }
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
    clearCanvas();
    const postResponse = await axios.post('/gen-image', {type:'mask', image: 'none'});
    if (postResponse.status===200){
      const getResponse = await axios.get('/mask-url');
      canvaseLoading.value = false;
      url.value = getResponse.data.url + '?' + Date.now();
      // const img = new Image();
      // img.src = getResponse.data.url;
      // img.onload = () => {
      //   context.value.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);
      // };
    } else{
      canvaseLoading.value = false;
      console.error('POST request failed');
    }
  } catch (error) {
    console.error(error);
  }
};
const sendMessage = ()=>{
  emitMessage('changeMessage', true)
}
const generateImage = async() => {
  try{
    const canvasImg = canvas.value.toDataURL('image/mask.png');
    sendMessage()
    const response = await axios.post('/gen-image', {type:'final', image: canvasImg});
  }catch(error){
    console.error(error);
  }
};
const clearCanvas = ()=>{
  context.value.clearRect(0,0,canvas.value.width, canvas.value.height);
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
    margin-top: 18px;
    width: 100px;
    height: 40px;
    font-size: 20px;
    color: rgb(58, 58, 58);
    margin-left: 12px;
    margin-right: 12px;
  }
  .brush-title{
    font-size: 12px;
    margin: 8px;
  }
  .brush-tool-box{
    width: 256px;
    height: 216px;
    margin-top: 12px;
    padding-bottom: 8px;
    border-radius: 16px;
    border: 2px #cecece dashed;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: top;
    background-color: white;
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
    margin-top: 12px;
    width: 200px;
    font-size: 18px;
  }
  .selectBar{
    margin-top: 8px;
    margin-bottom: 4px;
    width: 250px;
    /* display: flex;
    justify-content: center; */
  }
  .error-img{
    display: block;
    width: 32px;
    height:32px;
    margin-top: 112px;
    margin-left: 112px;
  }
</style>