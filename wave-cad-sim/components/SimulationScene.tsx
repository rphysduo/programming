"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

type LayerState = { cad: boolean; water: boolean; rays: boolean; board: boolean };

type SimulationProps = {
  playing: boolean;
  amplitudeMm: number;
  frequencyHz: number;
  phaseSpeed: number;
  rayDensity: number;
  layers: LayerState;
  viewRevision: number;
};

type SceneRuntime = {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  cad: THREE.Group;
  water: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  rays: THREE.Group;
  board: THREE.Group;
  animation: number;
  elapsed: number;
};

const WATER_Y = 0.185;

export default function SimulationScene(props: SimulationProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<SceneRuntime | null>(null);
  const propsRef = useRef(props);
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    propsRef.current = props;
  }, [props]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x061014);
    scene.fog = new THREE.FogExp2(0x061014, 0.72);

    const camera = new THREE.PerspectiveCamera(39, host.clientWidth / host.clientHeight, 0.01, 20);
    camera.position.set(1.52, 1.04, 1.58);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.18, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.minDistance = 0.65;
    controls.maxDistance = 4.2;
    controls.maxPolarAngle = Math.PI * 0.48;

    scene.add(new THREE.HemisphereLight(0xbcecff, 0x0a1518, 2.3));
    const keyLight = new THREE.DirectionalLight(0xc8f7ff, 3.3);
    keyLight.position.set(-1, 2.1, 1.4);
    scene.add(keyLight);
    const rimLight = new THREE.PointLight(0x22d3ee, 5.2, 3.2);
    rimLight.position.set(0.7, 0.5, -0.8);
    scene.add(rimLight);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(5.2, 5.2),
      new THREE.MeshStandardMaterial({ color: 0x061216, roughness: 0.98, metalness: 0.05 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.006;
    scene.add(floor);

    const board = new THREE.Group();
    const grid = new THREE.GridHelper(3.2, 32, 0x2a6670, 0x17343a);
    grid.material.transparent = true;
    grid.material.opacity = 0.52;
    board.add(grid);
    const axis = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.55, 0.004, 0), new THREE.Vector3(1.55, 0.004, 0)]),
      new THREE.LineBasicMaterial({ color: 0x6ee7f5, transparent: true, opacity: 0.52 }),
    );
    board.add(axis);
    scene.add(board);

    const cad = new THREE.Group();
    scene.add(cad);
    new STLLoader().load(
      "/full-assembly.stl",
      (geometry) => {
        geometry.rotateX(-Math.PI / 2);
        geometry.computeBoundingBox();
        const box = geometry.boundingBox!;
        const center = box.getCenter(new THREE.Vector3());
        geometry.translate(-center.x, -box.min.y, -center.z);
        geometry.computeVertexNormals();
        const shell = new THREE.Mesh(
          geometry,
          new THREE.MeshPhysicalMaterial({
            color: 0xa6bdc4,
            metalness: 0.34,
            roughness: 0.52,
            transparent: true,
            opacity: 0.24,
            transmission: 0.04,
            depthWrite: false,
            side: THREE.DoubleSide,
          }),
        );
        shell.renderOrder = 1;
        cad.add(shell);
        setLoadProgress(100);
      },
      (event) => setLoadProgress(event.total ? Math.round((event.loaded / event.total) * 100) : 34),
      () => setLoadError(true),
    );

    const waterMaterial = new THREE.ShaderMaterial({
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uAmplitude: { value: propsRef.current.amplitudeMm / 1000 },
        uFrequency: { value: propsRef.current.frequencyHz },
        uSpeed: { value: propsRef.current.phaseSpeed },
        uColorDeep: { value: new THREE.Color(0x057e91) },
        uColorCrest: { value: new THREE.Color(0x80f5ee) },
      },
      vertexShader: `
        uniform float uTime;
        uniform float uAmplitude;
        uniform float uFrequency;
        uniform float uSpeed;
        varying float vElevation;
        varying vec3 vWorldPosition;
        void main() {
          vec3 p = position;
          float k = 6.2831853 * uFrequency / max(uSpeed, 0.08);
          float phase = k * (p.x - uSpeed * uTime);
          float envelope = 0.84 + 0.16 * cos(p.z * 12.0);
          float elevation = uAmplitude * (sin(phase) + 0.22 * sin(phase * 2.0 + 0.75)) * envelope;
          p.y += elevation;
          vElevation = elevation / max(uAmplitude, 0.001);
          vec4 world = modelMatrix * vec4(p, 1.0);
          vWorldPosition = world.xyz;
          gl_Position = projectionMatrix * viewMatrix * world;
        }
      `,
      fragmentShader: `
        uniform vec3 uColorDeep;
        uniform vec3 uColorCrest;
        varying float vElevation;
        varying vec3 vWorldPosition;
        void main() {
          float crest = smoothstep(-0.8, 1.1, vElevation);
          float gridX = smoothstep(0.94, 1.0, abs(sin(vWorldPosition.x * 72.0)));
          float gridZ = smoothstep(0.94, 1.0, abs(sin(vWorldPosition.z * 76.0)));
          vec3 color = mix(uColorDeep, uColorCrest, crest * 0.72);
          color += (gridX + gridZ) * vec3(0.12, 0.34, 0.36);
          gl_FragColor = vec4(color, 0.72);
        }
      `,
    });
    const waterGeometry = new THREE.PlaneGeometry(1.42, 0.30, 150, 34);
    waterGeometry.rotateX(-Math.PI / 2);
    const water = new THREE.Mesh(waterGeometry, waterMaterial);
    water.position.set(-0.03, WATER_Y, 0.02);
    water.renderOrder = 3;
    scene.add(water);

    const rays = new THREE.Group();
    const cameraA = makeCamera(0xffa928);
    const cameraB = makeCamera(0xf15bb5);
    cameraA.position.set(-0.31, 0.53, 0.31);
    cameraB.position.set(-0.274, 0.53, 0.31);
    const aim = new THREE.Vector3(0.08, WATER_Y, 0);
    cameraA.lookAt(aim);
    cameraB.lookAt(aim);
    rays.add(cameraA, cameraB);
    const rayLines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.48, blending: THREE.AdditiveBlending }),
    );
    rayLines.renderOrder = 5;
    rays.add(rayLines);
    scene.add(rays);

    const clock = new THREE.Clock();
    const runtime: SceneRuntime = { renderer, camera, controls, cad, water, rays, board, animation: 0, elapsed: 0 };
    runtimeRef.current = runtime;

    const animate = () => {
      const current = propsRef.current;
      const delta = Math.min(clock.getDelta(), 0.05);
      if (current.playing) runtime.elapsed += delta;
      waterMaterial.uniforms.uTime.value = runtime.elapsed;
      waterMaterial.uniforms.uAmplitude.value = current.amplitudeMm / 1000;
      waterMaterial.uniforms.uFrequency.value = current.frequencyHz;
      waterMaterial.uniforms.uSpeed.value = current.phaseSpeed;
      cad.visible = current.layers.cad;
      water.visible = current.layers.water;
      rays.visible = current.layers.rays;
      board.visible = current.layers.board;
      updateRayGeometry(rayLines, current, runtime.elapsed);
      controls.update();
      renderer.render(scene, camera);
      runtime.animation = requestAnimationFrame(animate);
    };
    animate();

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);

    return () => {
      resizeObserver.disconnect();
      cancelAnimationFrame(runtime.animation);
      controls.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime || props.viewRevision === 0) return;
    runtime.camera.position.set(1.52, 1.04, 1.58);
    runtime.controls.target.set(0, 0.18, 0);
    runtime.controls.update();
  }, [props.viewRevision]);

  return (
    <div className="scene-host" ref={hostRef}>
      {loadProgress < 100 && !loadError && (
        <div className="model-loader">
          <span>IMPORTING CAD</span>
          <div><i style={{ width: `${Math.max(4, loadProgress)}%` }} /></div>
          <strong>{loadProgress}%</strong>
        </div>
      )}
      {loadError && <div className="model-loader error"><strong>CAD import failed</strong></div>}
      <div className="scene-hint">DRAG TO ORBIT · SCROLL TO ZOOM</div>
      <div className="camera-tag tag-a"><i />CAM A</div>
      <div className="camera-tag tag-b"><i />CAM B</div>
    </div>
  );
}

function makeCamera(color: number) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.056, 0.038, 0.07),
    new THREE.MeshStandardMaterial({ color: 0x122329, metalness: 0.75, roughness: 0.3 }),
  );
  group.add(body);
  const lens = new THREE.Mesh(
    new THREE.CylinderGeometry(0.014, 0.018, 0.035, 18),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.4 }),
  );
  lens.rotation.x = Math.PI / 2;
  lens.position.z = 0.048;
  group.add(lens);
  const frustum = new THREE.Mesh(
    new THREE.ConeGeometry(0.072, 0.16, 4, 1, true),
    new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.18 }),
  );
  frustum.rotation.x = Math.PI / 2;
  frustum.position.z = 0.14;
  group.add(frustum);
  return group;
}

function updateRayGeometry(lines: THREE.LineSegments, props: SimulationProps, time: number) {
  const positions: number[] = [];
  const colors: number[] = [];
  const cameraA = new THREE.Vector3(-0.31, 0.53, 0.31);
  const cameraB = new THREE.Vector3(-0.274, 0.53, 0.31);
  const colorA = new THREE.Color(0xffa928);
  const colorB = new THREE.Color(0xf15bb5);
  const amplitude = props.amplitudeMm / 1000;
  const k = (Math.PI * 2 * props.frequencyHz) / Math.max(props.phaseSpeed, 0.08);
  for (let index = 0; index < props.rayDensity; index += 1) {
    const t = props.rayDensity === 1 ? 0.5 : index / (props.rayDensity - 1);
    const x = -0.54 + t * 1.08;
    const z = -0.1 + (index % 3) * 0.1;
    const phase = k * (x - props.phaseSpeed * time);
    const y = WATER_Y + amplitude * (Math.sin(phase) + 0.22 * Math.sin(phase * 2 + 0.75));
    const target = new THREE.Vector3(x, y, z);
    pushSegment(positions, colors, cameraA, target, colorA);
    pushSegment(positions, colors, cameraB, target, colorB);
  }
  lines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  lines.geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  lines.geometry.computeBoundingSphere();
}

function pushSegment(positions: number[], colors: number[], start: THREE.Vector3, end: THREE.Vector3, color: THREE.Color) {
  positions.push(start.x, start.y, start.z, end.x, end.y, end.z);
  colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
}
