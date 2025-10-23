int W = 600;  // 宽度改为600
int H = 800;  // 高度改为800，形成4:3竖版

ArrayList<Snow> snows;
ArrayList<Particle> particles;
int maxSnows = 40; // 减少数量以平衡更大的雪花
color[] snowColors = {
  color(200, 230, 255), // 浅蓝1
  color(180, 220, 255), // 浅蓝2
  color(160, 210, 255), // 浅蓝3
  color(190, 225, 255)  // 浅蓝4
};

void setup() {
  size(600, 800);  // 4:3竖版画布
  smooth();
  
  snows = new ArrayList<Snow>();
  particles = new ArrayList<Particle>();
  
  // 初始化雪花 - 在整个画布上方随机分布
  for (int i = 0; i < maxSnows; i++) {
    float size = random(15, 60); // 从3-12变为15-60
    int type = (int)random(4); // 0-3 四种雪花类型
    // 在整个画布宽度上随机分布
    float startX = random(0, width);
    float startY = random(-200, 0);
    snows.add(new Snow(startX, startY, size, type));
  }
}

void draw() {
  // 纯白背景
  background(255);
  
  // 更新和显示雪花
  for (int i = snows.size() - 1; i >= 0; i--) {
    Snow s = snows.get(i);
    s.applyWind();
    s.update();
    s.display();
    
    if (s.offScreen()) {
      // 当雪花离开屏幕时，从顶部重新生成，位置在整个画布宽度上随机
      float newX = random(0, width);
      snows.set(i, new Snow(newX, random(-200, -50), random(15, 60), (int)random(4)));
    }
  }
  
  // 更新和显示粒子
  for (int i = particles.size() - 1; i >= 0; i--) {
    Particle p = particles.get(i);
    p.update();
    p.display();
    if (p.dead()) {
      particles.remove(i);
    }
  }
  
  // 显示交互提示
  drawInstructions();
}

void drawInstructions() {
  fill(0, 150);
  textSize(16);
  textAlign(LEFT, TOP);
  text("点击雪花使其弥散", 20, 20);
}

void mousePressed() {
  // 检查是否点击到雪花
  for (int i = snows.size() - 1; i >= 0; i--) {
    Snow s = snows.get(i);
    if (s.contains(mouseX, mouseY)) {
      // 创建弥散粒子 - 根据雪花类型创建不同数量的粒子
      int particleCount = 25 + (int)(s.size * 0.4); // 调整粒子数量
      for (int k = 0; k < particleCount; k++) {
        // 粒子从雪花中心向外弥散
        float angle = random(TWO_PI);
        float speed = random(2, 8); // 速度增加
        float px = s.x + cos(angle) * s.size;
        float py = s.y + sin(angle) * s.size;
        particles.add(new Particle(px, py, cos(angle) * speed, sin(angle) * speed, random(2, 8), s.col)); // 粒子大小增加
      }
      
      // 替换雪花 - 从顶部随机位置重新生成
      snows.set(i, new Snow(random(0, width), random(-200, -50), random(15, 60), (int)random(4)));
      break;
    }
  }
}

class Snow {
  float x, y, size;
  float vy, vx;
  float driftPhase;
  float rotation;
  float rotationSpeed;
  int type;
  color col;
  
  Snow(float x_, float y_, float size_, int type_) {
    x = x_;
    y = y_;
    size = size_;
    type = type_;
    
    // 根据大小设置下落速度 - 速度相应增加
    vy = map(size, 15, 60, 1.5, 7.5) * random(0.8, 1.2); // 从0.3-1.5变为1.5-7.5
    // 随机横向速度，有些向左有些向右，但整体略微向左
    vx = random(-1.5, 0.5); // 随机横向速度
    
    driftPhase = random(TWO_PI);
    rotation = random(TWO_PI);
    rotationSpeed = random(-0.01, 0.01); // 旋转速度稍微减慢
    
    // 随机选择一种浅蓝色
    col = snowColors[(int)random(snowColors.length)];
  }
  
  void applyWind() {
    driftPhase += 0.005;
    float wind = sin(driftPhase) * 0.8;
    vx += wind * 0.002;
    vx *= 0.998; // 轻微阻尼
  }
  
  void update() {
    // 添加自然的飘动
    float n = noise(x * 0.003, y * 0.003, frameCount * 0.003);
    float sway = map(n, 0, 1, -0.8, 0.8);
    x += vx + sway * 0.5;
    y += vy;
    
    // 旋转
    rotation += rotationSpeed;
  }
  
  void display() {
    pushMatrix();
    translate(x, y);
    rotate(rotation);
    
    noStroke();
    fill(col, 180); // 半透明
    
    // 根据类型绘制不同形状的雪花
    switch(type) {
      case 0: drawHexagonalSnowflake(size); break; // 六角形
      case 1: drawStellarSnowflake(size); break;   // 星形
      case 2: drawFernSnowflake(size); break;      // 蕨类形
      case 3: drawDendriticSnowflake(size); break; // 树枝形
    }
    
    popMatrix();
  }
  
  // 六角形雪花
  void drawHexagonalSnowflake(float size) {
    for (int i = 0; i < 6; i++) {
      float angle = i * PI/3;
      // 主枝
      drawSnowflakeArm(0, 0, angle, size, 3.0); // 线条粗细增加
      // 侧枝
      drawSnowflakeArm(size*0.3, 0, angle + PI/6, size*0.5, 2.0);
      drawSnowflakeArm(size*0.3, 0, angle - PI/6, size*0.5, 2.0);
    }
  }
  
  // 星形雪花
  void drawStellarSnowflake(float size) {
    for (int i = 0; i < 8; i++) {
      float angle = i * PI/4;
      drawSnowflakeArm(0, 0, angle, size*1.2, 3.0);
      // 更多的侧枝
      for (int j = 1; j <= 2; j++) {
        float pos = size * 0.2 * j;
        drawSnowflakeArm(pos, 0, angle + PI/8, size*0.4, 1.8);
        drawSnowflakeArm(pos, 0, angle - PI/8, size*0.4, 1.8);
      }
    }
  }
  
  // 蕨类形雪花
  void drawFernSnowflake(float size) {
    for (int i = 0; i < 6; i++) {
      float angle = i * PI/3;
      // 主枝
      drawSnowflakeArm(0, 0, angle, size, 3.0);
      // 蕨类侧枝
      drawFernBranch(size*0.4, 0, angle + PI/12, size*0.6, 2.4, 3);
      drawFernBranch(size*0.4, 0, angle - PI/12, size*0.6, 2.4, 3);
    }
  }
  
  // 树枝形雪花
  void drawDendriticSnowflake(float size) {
    for (int i = 0; i < 6; i++) {
      float angle = i * PI/3;
      // 主枝
      drawSnowflakeArm(0, 0, angle, size, 3.0);
      // 多级分支
      drawBranch(size*0.3, 0, angle + PI/6, size*0.7, 2.1, 2);
      drawBranch(size*0.3, 0, angle - PI/6, size*0.7, 2.1, 2);
    }
  }
  
  // 绘制雪花臂
  void drawSnowflakeArm(float x, float y, float angle, float length, float thickness) {
    pushMatrix();
    translate(x, y);
    rotate(angle);
    
    // 绘制臂
    float endX = length;
    stroke(col, 200);
    strokeWeight(thickness);
    line(0, 0, endX, 0);
    
    // 臂端的小圆点
    noStroke();
    fill(col, 220);
    ellipse(endX, 0, thickness*2, thickness*2);
    
    popMatrix();
  }
  
  // 绘制蕨类分支
  void drawFernBranch(float x, float y, float angle, float length, float thickness, int levels) {
    if (levels <= 0) return;
    
    pushMatrix();
    translate(x, y);
    rotate(angle);
    
    // 绘制分支
    float endX = length;
    stroke(col, 200);
    strokeWeight(thickness);
    line(0, 0, endX, 0);
    
    // 递归绘制更小的分支
    if (levels > 1) {
      drawFernBranch(endX*0.3, 0, PI/8, length*0.6, thickness*0.7, levels-1);
      drawFernBranch(endX*0.3, 0, -PI/8, length*0.6, thickness*0.7, levels-1);
      drawFernBranch(endX*0.6, 0, PI/12, length*0.4, thickness*0.5, levels-1);
      drawFernBranch(endX*0.6, 0, -PI/12, length*0.4, thickness*0.5, levels-1);
    }
    
    popMatrix();
  }
  
  // 绘制树枝分支
  void drawBranch(float x, float y, float angle, float length, float thickness, int levels) {
    if (levels <= 0) return;
    
    pushMatrix();
    translate(x, y);
    rotate(angle);
    
    // 绘制分支
    float endX = length;
    stroke(col, 200);
    strokeWeight(thickness);
    line(0, 0, endX, 0);
    
    // 分支末端的小圆点
    noStroke();
    fill(col, 220);
    ellipse(endX, 0, thickness*2, thickness*2);
    
    // 递归绘制更小的分支
    if (levels > 1) {
      drawBranch(endX, 0, PI/6, length*0.6, thickness*0.7, levels-1);
      drawBranch(endX, 0, -PI/6, length*0.6, thickness*0.7, levels-1);
    }
    
    popMatrix();
  }
  
  boolean offScreen() {
    // 只检查是否超出底部，不检查左侧
    return (y - size > height);
  }
  
  boolean contains(float px, float py) {
    return dist(px, py, x, y) <= size * 1.5; // 点击区域稍微增大
  }
}

class Particle {
  float x, y, vx, vy, size;
  float life, maxLife;
  color col;
  float rotation;
  float rotationSpeed;
  
  Particle(float x_, float y_, float vx_, float vy_, float size_, color col_) {
    x = x_;
    y = y_;
    vx = vx_;
    vy = vy_;
    size = size_;
    maxLife = random(40, 100);
    life = maxLife;
    col = col_;
    rotation = random(TWO_PI);
    rotationSpeed = random(-0.1, 0.1);
  }
  
  void update() {
    // 物理模拟
    vy += 0.08; // 重力
    vx *= 0.98; // 空气阻力
    vy *= 0.98;
    x += vx;
    y += vy;
    life -= 1;
    rotation += rotationSpeed;
    
    // 尺寸逐渐缩小
    size *= 0.99;
  }
  
  void display() {
    float alpha = map(life, 0, maxLife, 0, 200);
    
    pushMatrix();
    translate(x, y);
    rotate(rotation);
    
    noStroke();
    fill(red(col), green(col), blue(col), alpha);
    
    // 绘制粒子 - 使用简单形状
    if (size > 3) {
      // 较大的粒子绘制为小雪花碎片
      ellipse(0, 0, size*2, size*1.5);
      ellipse(0, 0, size*1.5, size*2);
    } else {
      // 较小的粒子绘制为圆点
      ellipse(0, 0, size*2, size*2);
    }
    
    popMatrix();
  }
  
  boolean dead() {
    return life <= 0 || y - size > height + 50 || size < 0.1;
  }
}
