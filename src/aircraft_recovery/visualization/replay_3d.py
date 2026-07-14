"""Standalone Panda3D replay viewer; never feeds data back to the simulator."""
from __future__ import annotations
import json
import math
from bisect import bisect_right
from datetime import datetime,timezone
from pathlib import Path
from typing import Any, Literal
import warnings
import yaml
from pydantic import BaseModel, ConfigDict, Field

class AircraftModelSettings(BaseModel):
    model_config=ConfigDict(extra="forbid")
    model_path:str="assets/aircraft/cessna/cessna.fbx"
    scale:float=Field(default=0.00056,gt=0)
    pitch_offset_deg:float=90.0
    yaw_offset_deg:float=180.0
    roll_offset_deg:float=0.0
    position_offset_m:tuple[float,float,float]=(0.0,0.0,0.0)
    neutral_color_rgba:tuple[float,float,float,float]=(0.72,0.74,0.78,1.0)

class CameraSettings(BaseModel):
    model_config=ConfigDict(extra="forbid")
    chase_distance_m:float=Field(default=18.0,gt=0)
    chase_height_m:float=6.0
    look_ahead_m:float=6.0
    lateral_offset_m:float=0.0
    smoothing:float=Field(default=8.0,gt=0)
    min_distance_m:float=Field(default=6.0,gt=0)
    max_distance_m:float=Field(default=40.0,gt=0)
    zoom_speed_m:float=Field(default=2.0,gt=0)
    orbit_sensitivity_x:float=Field(default=.18,gt=0)
    orbit_sensitivity_y:float=Field(default=.14,gt=0)
    orbit_pitch_min_deg:float=-25.0
    orbit_pitch_max_deg:float=65.0
    free_move_speed_mps:float=Field(default=20.0,gt=0)
    free_fast_multiplier:float=Field(default=4.0,gt=1)
    field_of_view_deg:float=Field(default=62.0,gt=10,lt=150)

class WorldSettings(BaseModel):
    model_config=ConfigDict(extra="forbid")
    runway_length_m:float=Field(default=1200.0,gt=0)
    runway_width_m:float=Field(default=30.0,gt=0)
    path_altitude_offset_m:float=0.5
    horizontal_scale:float=Field(default=1.0,gt=0)
    vertical_scale:float=Field(default=0.12,gt=0)
    ground_size_m:float=Field(default=12000.0,gt=100)
    grid_spacing_m:float=Field(default=500.0,gt=0)

class Replay3DSettings(BaseModel):
    model_config=ConfigDict(extra="forbid")
    aircraft:AircraftModelSettings=AircraftModelSettings()
    camera:CameraSettings=CameraSettings()
    world:WorldSettings=WorldSettings()

def load_replay_3d_settings(path:str|Path)->Replay3DSettings:
    with Path(path).open("r",encoding="utf-8") as stream: data=yaml.safe_load(stream)
    return Replay3DSettings.model_validate(data)

def replay_parent_hpr(heading_deg:float,pitch_deg:float,roll_deg:float)->tuple[float,float,float]:
    """Map aviation heading/pitch/right-bank into Panda3D H/P/R."""
    return (-heading_deg,pitch_deg,roll_deg)

def local_xy_m(latitude:float,longitude:float,origin_latitude:float,origin_longitude:float)->tuple[float,float]:
    north=(latitude-origin_latitude)*60.0*1852.0
    east=(longitude-origin_longitude)*60.0*1852.0*math.cos(math.radians(origin_latitude))
    return east,north

def _short_heading_lerp(a:float,b:float,t:float)->float:
    return (a+((b-a+180.0)%360.0-180.0)*t)%360.0

def load_replay_frames(run_directory:str|Path)->tuple[list[dict[str,float]],dict[str,Any],dict[str,Any]]:
    run=Path(run_directory)
    if not run.is_dir(): raise FileNotFoundError(f"Replay run directory does not exist: {run}")
    rows=[json.loads(line) for line in (run/"history.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows: raise ValueError("3D replay requires at least one history row")
    frames=[]
    for row in rows:
        truth=row.get("ground_truth_after",{}).get("true_state",{})
        observed=row["next_observation"]["aircraft_state"]
        navigation=row["next_observation"]["navigation"]
        frames.append({
            "time":float(row["next_observation"]["timestamp_seconds"]),
            "latitude":float(truth.get("latitude_deg",navigation["latitude_deg"])),
            "longitude":float(truth.get("longitude_deg",navigation["longitude_deg"])),
            "altitude_ft":float(truth.get("altitude_ft",observed["altitude_ft"])),
            "heading_deg":float(truth.get("heading_deg",observed["heading_deg"])),
            "pitch_deg":float(truth.get("pitch_deg",observed["pitch_deg"])),
            "roll_deg":float(truth.get("roll_deg",observed["roll_deg"])),
            "airspeed_kts":float(truth.get("airspeed_kts",observed["airspeed_kts"])),
        })
    return frames,json.loads((run/"summary.json").read_text(encoding="utf-8")),json.loads((run/"metadata.json").read_text(encoding="utf-8"))

class Replay3DViewer:
    def __init__(self,run_directory:str|Path,settings:Replay3DSettings,camera_mode:Literal["chase","fixed","top"]="chase",speed:float=1.0,show_path:bool=False,window_type:str|None=None,show_failures:bool=False,show_controller_known_failures:bool=False,show_ground_truth_failures:bool=False,start_paused:bool=False,debug_coordinates:bool=False)->None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("","window-title Aircraft Recovery 3D Replay\nframebuffer-multisample 1\nmultisamples 4\nsync-video 1")
        from direct.showbase.ShowBase import ShowBase
        self.base=ShowBase(windowType=window_type)
        self.base.disableMouse()
        self.run_directory=Path(run_directory); self.settings=settings; self.camera_mode=camera_mode; self.speed=speed; self.show_path=show_path; self.show_failures=show_failures; self.show_known=show_controller_known_failures or show_failures; self.show_ground_truth=show_ground_truth_failures; self.show_monitor_details=False; self.show_help=False
        self.frames,self.summary,self.metadata=load_replay_frames(run_directory)
        from aircraft_recovery.visualization.replay_events import ReplayEventState,ReplayTimelineEvent,normalize_replay_data
        self.replay_data=normalize_replay_data(run_directory)
        monitor_timeline=[ReplayTimelineEvent(f"monitor:{i}",event.time_s,"monitor",(event.status,),monitor=event) for i,event in enumerate(self.replay_data.monitor_events)]
        self.event_state=ReplayEventState(self.replay_data.failure_events+monitor_timeline)
        self.session_path=self._start_session_log()
        self.origin_lat=self.frames[0]["latitude"]; self.origin_lon=self.frames[0]["longitude"]
        airports=(self.metadata.get("scenario_configuration") or {}).get("airports") or []
        self.ground_elevation_ft=float(airports[0].get("elevation_ft",0.0)) if airports else 0.0
        world=settings.world
        self.points=[]
        for frame in self.frames:
            east,north=local_xy_m(frame["latitude"],frame["longitude"],self.origin_lat,self.origin_lon)
            z=max(0.0,(frame["altitude_ft"]-self.ground_elevation_ft)*0.3048*world.vertical_scale)+world.path_altitude_offset_m
            self.points.append((east*world.horizontal_scale,north*world.horizontal_scale,z))
        if debug_coordinates: print(f"Replay world coordinates: first={self.points[0]} last={self.points[-1]}")
        self.aircraft_parent=self.base.render.attachNewNode("aircraft_replay_transform")
        self.model_node,self.model_loaded,self.model_message=self._load_aircraft_child()
        self._log_event("model_loaded" if self.model_loaded else "model_fallback",0.0,message=self.model_message)
        self._setup_world(); self._setup_lighting(); self._setup_failure_markers(); self._setup_ui()
        if self.base.camLens is not None:
            self.base.camLens.setFov(settings.camera.field_of_view_deg); self.base.camLens.setNearFar(.1,30000)
        from panda3d.core import ClockObject
        self.elapsed=0.0; self.paused=True; self.start_paused=start_paused; self.frame_index=0; self.path_drawn_index=0; self.notification_until=0.0; self.camera_distance=settings.camera.chase_distance_m; self.camera_distance_target=self.camera_distance; self.camera_initialized=False
        self.orbit_yaw_deg=0.0; self.orbit_pitch_deg=0.0; self.mouse_look_active=False; self.free_move_speed=settings.camera.free_move_speed_mps; self.key_state={name:False for name in ("forward","back","left","right","down","up","fast")}
        self.camera_mode={"fixed":"fixed","top":"top"}.get(camera_mode,"chase")
        self.auto_start_at=None if start_paused else ClockObject.getGlobalClock().getRealTime()+2.0
        self.base.accept("escape",self.base.userExit); self.base.accept("space",self._toggle_pause); self.base.accept("r",self._restart); self.base.accept("f",self._handle_focus); self.base.accept("g",self._toggle_ground_truth); self.base.accept("m",self._toggle_monitor); self.base.accept("h",self._toggle_help); self.base.accept("t",self._toggle_path); self.base.accept("c",self._reset_camera); self.base.accept("arrow_left",self._step,[-1]); self.base.accept("arrow_right",self._step,[1]); self.base.accept("wheel_up",self._zoom,[-1]); self.base.accept("wheel_down",self._zoom,[1]); self.base.accept("mouse3",self._start_mouse_look); self.base.accept("mouse3-up",self._stop_mouse_look)
        for key,mode in (("1","chase"),("2","side"),("3","top"),("4","free"),("5","cockpit"),("6","fixed")): self.base.accept(key,self._set_camera_mode,[mode])
        for key,name in (("w","forward"),("s","back"),("a","left"),("d","right"),("q","down"),("e","up"),("shift","fast")):
            self.base.accept(key,self._set_key,[name,True]); self.base.accept(f"{key}-up",self._set_key,[name,False])
        self.base.taskMgr.add(self._update,"replay-update")

    def _load_aircraft_child(self):
        from panda3d.core import Filename,Material
        cfg=self.settings.aircraft; path=Path(cfg.model_path).resolve()
        try:
            if not path.is_file(): raise FileNotFoundError(path)
            model=self.base.loader.loadModel(Filename.fromOsSpecific(str(path)),noCache=True)
            if model.isEmpty(): raise RuntimeError("Panda3D returned an empty model")
            model.reparentTo(self.aircraft_parent); model.setScale(cfg.scale); model.setHpr(cfg.yaw_offset_deg,cfg.pitch_offset_deg,cfg.roll_offset_deg); model.setPos(*cfg.position_offset_m)
            if model.findAllTextures().getNumTextures()==0:
                material=Material("neutral-untextured-aircraft"); material.setAmbient((.36,.38,.41,1)); material.setDiffuse(cfg.neutral_color_rgba); material.setSpecular((.32,.34,.38,1)); material.setShininess(22); model.setMaterial(material,1); model.setColor(*cfg.neutral_color_rgba); model.setTransparency(False); model.setTwoSided(False)
                message=f"Loaded FBX without textures; neutral material applied: {path}"
            else: message=f"Loaded FBX aircraft model: {path}"
            return model,True,message
        except Exception as error:
            message=f"Could not load FBX '{path}' ({type(error).__name__}: {error}); using primitive fallback aircraft."
            warnings.warn(message,RuntimeWarning,stacklevel=2)
            return self._create_primitive_aircraft(),False,message

    def _create_primitive_aircraft(self):
        root=self.aircraft_parent.attachNewNode("primitive_fallback_aircraft")
        def cube(name,pos,scale,color):
            node=self.base.loader.loadModel("models/misc/rgbCube"); node.setName(name); node.reparentTo(root); node.setPos(*pos); node.setScale(*scale); node.setColor(*color); return node
        neutral=(.72,.74,.78,1); cube("fuselage",(0,0,0),(0.65,4.0,0.65),neutral); cube("nose",(0,4.2,0),(0.8,1.0,0.75),(.82,.84,.88,1)); cube("wing",(0,.2,.15),(5.2,.65,.12),neutral); cube("tailplane",(0,-3.2,.45),(2.2,.45,.10),neutral); cube("fin",(0,-3.3,1.05),(.10,.55,1.0),neutral)
        return root

    def _setup_world(self):
        from panda3d.core import CardMaker,Fog,LineSegs,TransparencyAttrib
        size=self.settings.world.ground_size_m/2
        ground=CardMaker("ground"); ground.setFrame(-size,size,-size,size); g=self.base.render.attachNewNode(ground.generate()); g.setP(-90); g.setColor(.16,.23,.18,1); g.setTwoSided(True)
        grid=LineSegs("ground-grid"); grid.setThickness(1); grid.setColor(.42,.52,.45,.28); spacing=self.settings.world.grid_spacing_m
        coordinate=-size
        while coordinate<=size:
            grid.moveTo(coordinate,-size,.08); grid.drawTo(coordinate,size,.08); grid.moveTo(-size,coordinate,.08); grid.drawTo(size,coordinate,.08); coordinate+=spacing
        grid_node=self.base.render.attachNewNode(grid.create()); grid_node.setTransparency(TransparencyAttrib.MAlpha)
        scenario=self.metadata.get("scenario_configuration") or {}; airports=scenario.get("airports") or []
        if airports:
            airport=airports[0]; distance=airport["distance_nm"]*1852; bearing=math.radians(airport["bearing_deg"]); x=math.sin(bearing)*distance; y=math.cos(bearing)*distance
        else: x,y=0,1200
        runway=CardMaker("runway"); runway.setFrame(-self.settings.world.runway_width_m/2,self.settings.world.runway_width_m/2,-self.settings.world.runway_length_m/2,self.settings.world.runway_length_m/2); r=self.base.render.attachNewNode(runway.generate()); r.setP(-90); r.setPos(x,y,.05); r.setH(-(airports[0]["runway_heading_deg"] if airports else 0)); r.setColor(.18,.18,.2,1)
        self.path_node=None; self._draw_path(0)
        fog=Fog("horizon-fog"); fog.setColor(.45,.61,.72); fog.setLinearRange(2500,9000); self.base.render.setFog(fog)

    def _draw_path(self,index:int,pos=None):
        from panda3d.core import LineSegs,TransparencyAttrib
        if self.path_node is not None: self.path_node.removeNode()
        line=LineSegs("replay-trail"); line.setThickness(2.5); line.setColor(.1,.75,.9,.72); line.moveTo(*self.points[0])
        for point in self.points[1:index+1]: line.drawTo(*point)
        if pos is not None and index>0: line.drawTo(*pos)
        self.path_node=self.base.render.attachNewNode(line.create()); self.path_node.setTransparency(TransparencyAttrib.MAlpha)
        if not self.show_path: self.path_node.hide()

    def _setup_lighting(self):
        from panda3d.core import AmbientLight,DirectionalLight
        ambient=AmbientLight("ambient"); ambient.setColor((.48,.48,.52,1)); self.base.render.setLight(self.base.render.attachNewNode(ambient))
        sun=DirectionalLight("sun"); sun.setColor((.9,.88,.82,1)); node=self.base.render.attachNewNode(sun); node.setHpr(-30,-55,0); self.base.render.setLight(node); self.base.setBackgroundColor(.45,.68,.9,1)

    def _setup_ui(self):
        from direct.gui.DirectGui import DirectFrame,DirectWaitBar
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        panel=(.025,.035,.05,.82); border=(.18,.24,.3,1)
        def frame(parent,bounds,pos): return DirectFrame(parent=parent,frameColor=panel,frameSize=bounds,relief=1,borderWidth=(.006,.006),frameTexture=None,pos=pos)
        self.telemetry_frame=frame(self.base.a2dTopLeft,(0,.69,-.31,0),(.035,0,-.045)); self.failure_frame=frame(self.base.a2dTopRight,(-.76,0,-.84,0),(-.035,0,-.045)); self.timeline_frame=frame(self.base.a2dBottomLeft,(0,.82,0,.29),(.035,0,.04)); self.monitor_frame=frame(self.base.a2dBottomRight,(-.82,0,0,.36),(-.035,0,.04))
        self.telemetry=OnscreenText(parent=self.telemetry_frame,text="",pos=(.025,-.045),scale=.032,align=TextNode.ALeft,mayChange=True,fg=(.92,.96,1,1),wordwrap=34)
        self.failure_panel=OnscreenText(parent=self.failure_frame,text="",pos=(-.73,-.04),scale=.027,align=TextNode.ALeft,mayChange=True,fg=(1,.94,.82,1),wordwrap=43)
        self.monitor_panel=OnscreenText(parent=self.monitor_frame,text="",pos=(-.78,.315),scale=.027,align=TextNode.ALeft,mayChange=True,fg=(.82,.94,1,1),wordwrap=48)
        self.timeline_panel=OnscreenText(parent=self.timeline_frame,text="",pos=(.025,.255),scale=.026,align=TextNode.ALeft,mayChange=True,fg=(.9,.93,.95,1),wordwrap=55)
        self.replay_notice=OnscreenText(parent=self.base.aspect2d,text="RECORDED SIMULATION REPLAY\nVISUALIZATION ONLY - NOT LIVE CONTROL",pos=(0,.95),scale=.024,align=TextNode.ACenter,fg=(1,.9,.55,1),shadow=(0,0,0,1))
        self.notification_frame=DirectFrame(parent=self.base.aspect2d,frameColor=(.04,.02,.02,.9),frameSize=(-.58,.58,-.23,.23),pos=(0,0,.45)); self.notification=OnscreenText(parent=self.notification_frame,text="",pos=(0,.1),scale=.048,align=TextNode.ACenter,mayChange=True,fg=(1,.28,.18,1),wordwrap=28); self.notification_frame.hide()
        self.help_frame=frame(self.base.aspect2d,(-.55,.55,-.55,.55),(0,0,0)); self.help_panel=OnscreenText(parent=self.help_frame,text="CAMERA / REPLAY CONTROLS\n\nMouse wheel  Smooth zoom\nRight mouse drag  Orbit / mouse look\nC  Reset chase camera\n1 Chase   2 Side   3 Top\n4 Free    5 Cockpit   6 Ground\nWASD  Move free camera\nQ / E  Down / up\nShift  Faster free movement\nF  Focus aircraft (free)\n\nSpace  Pause / resume\nR  Restart replay\nH  Close controls\nEsc  Exit",pos=(-.49,.47),scale=.036,align=TextNode.ALeft,fg=(.95,.97,1,1)); self.help_frame.hide()
        self.health_bars={}
        OnscreenText(parent=self.failure_frame,text="CONTROL / ENGINE HEALTH",pos=(-.73,-.37),scale=.025,align=TextNode.ALeft,fg=(.75,.84,.9,1))
        self.health_labels={}
        for name,y,color in (("elevator",-.43,(.2,.72,.95,1)),("aileron",-.49,(1,.68,.16,1)),("rudder",-.55,(.2,.72,.95,1)),("left_engine",-.66,(.95,.25,.18,1)),("right_engine",-.72,(.25,.78,.46,1))):
            bar=DirectWaitBar(parent=self.failure_frame,range=100,value=100,frameSize=(-.19,.19,-.014,.014),pos=(-.245,0,y),frameColor=(.1,.12,.15,1),barColor=color,text="",relief=1); self.health_bars[name]=bar
            label={"elevator":"Elevator","aileron":"Aileron","rudder":"Rudder","left_engine":"Engine Left","right_engine":"Engine Right"}[name]
            self.health_labels[name]=OnscreenText(parent=self.failure_frame,text=label,pos=(-.73,y+.01),scale=.023,align=TextNode.ALeft,fg=(.9,.93,.95,1))

    def _setup_failure_markers(self):
        from panda3d.core import TextNode
        self.failure_markers={}
        for failure in self.replay_data.failures:
            text=failure.display_name.upper(); label_color=(1,.82,.55,1); label_align=TextNode.ACenter
            root=self.aircraft_parent.attachNewNode(f"marker-{failure.failure_id}"); marker=self.base.loader.loadModel("models/misc/sphere"); marker.reparentTo(root); marker.setScale(.24)
            if failure.subsystem=="left_engine": root.setPos(-2.0,2.2,.75); marker.setColor(1,.08,.04,1); text="LEFT ENGINE FAILED"; label_color=(1,.3,.2,1); label_align=TextNode.ARight
            elif "aileron" in failure.subsystem: root.setPos(2.2,0,.75); marker.setColor(1,.58,.05,1); text=f"AILERON AUTHORITY {self.replay_data.state_at(99e9).aileron*100:.0f}%"; label_align=TextNode.ALeft
            else: root.setPos(0,0,2)
            label=TextNode(f"label-{failure.failure_id}"); label.setText(text); label.setTextColor(*label_color); label.setAlign(label_align); label_path=root.attachNewNode(label); label_path.setScale(.26); label_path.setPos(0,0,.85); label_path.setBillboardPointEye(); root.hide(); self.failure_markers[failure.failure_id]=root

    def _start_session_log(self)->Path:
        directory=Path("results/replay_3d_sessions"); directory.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"); path=directory/f"{self.run_directory.name}_{stamp}.jsonl"
        with path.open("w",encoding="utf-8") as stream: stream.write(json.dumps({"event":"session_started","wall_time_utc":datetime.now(timezone.utc).isoformat(),"source_run":str(self.run_directory),"model_path":self.settings.aircraft.model_path,"camera_mode":self.camera_mode,"playback_speed":self.speed,"missing_fields":self.replay_data.missing_fields})+"\n")
        return path

    def _log_event(self,event:str,replay_time_s:float,**details):
        with self.session_path.open("a",encoding="utf-8") as stream: stream.write(json.dumps({"event":event,"replay_time_s":replay_time_s,"wall_time_utc":datetime.now(timezone.utc).isoformat(),**details})+"\n")

    def _toggle_pause(self): self.auto_start_at=None; self.paused=not self.paused
    def _restart(self):
        from panda3d.core import ClockObject
        self.elapsed=0.0; self.frame_index=0; self.camera_initialized=False; self.event_state.reset(); self.notification.setText(""); self.notification_frame.hide(); self.paused=True; self.auto_start_at=None if self.start_paused else ClockObject.getGlobalClock().getRealTime()+2.0; self._draw_path(0); self._log_event("restart",0.0)
    def _toggle_failures(self): self.show_failures=not self.show_failures
    def _toggle_ground_truth(self): self.show_ground_truth=not self.show_ground_truth
    def _toggle_monitor(self): self.show_monitor_details=not self.show_monitor_details
    def _toggle_help(self): self.show_help=not self.show_help; self.help_frame.show() if self.show_help else self.help_frame.hide()
    def _zoom(self,direction:float):
        cfg=self.settings.camera
        if self.camera_mode=="chase": self.camera_distance_target=max(cfg.min_distance_m,min(cfg.max_distance_m,self.camera_distance_target+direction*cfg.zoom_speed_m))
        elif self.camera_mode=="free": self.free_move_speed=max(1.0,min(250.0,self.free_move_speed*(1.2 if direction<0 else 1/1.2)))
    def _reset_camera(self):
        self.camera_mode="chase"; self.orbit_yaw_deg=0.0; self.orbit_pitch_deg=0.0; self.camera_distance=self.settings.camera.chase_distance_m; self.camera_distance_target=self.camera_distance; self.camera_initialized=False
    def _set_camera_mode(self,mode:str):
        self.camera_mode=mode
        if mode!="free": self.camera_initialized=False
    def _set_key(self,name:str,pressed:bool): self.key_state[name]=pressed
    def _handle_focus(self):
        if self.camera_mode=="free": self.base.camera.lookAt(self.aircraft_parent)
        else: self._toggle_failures()
    def _start_mouse_look(self):
        self.mouse_look_active=True
        if self.base.win is not None and hasattr(self.base.win,"requestProperties"):
            from panda3d.core import WindowProperties
            props=WindowProperties(); props.setCursorHidden(True); self.base.win.requestProperties(props); self._center_pointer()
    def _stop_mouse_look(self):
        self.mouse_look_active=False
        if self.base.win is not None and hasattr(self.base.win,"requestProperties"):
            from panda3d.core import WindowProperties
            props=WindowProperties(); props.setCursorHidden(False); self.base.win.requestProperties(props)
    def _center_pointer(self):
        if self.base.win is not None and hasattr(self.base.win,"movePointer"): self.base.win.movePointer(0,self.base.win.getXSize()//2,self.base.win.getYSize()//2)
    def _apply_mouse_delta(self,dx:float,dy:float):
        cfg=self.settings.camera
        if self.camera_mode=="free":
            hpr=self.base.camera.getHpr(); pitch=max(cfg.orbit_pitch_min_deg,min(cfg.orbit_pitch_max_deg,hpr.y-dy*cfg.orbit_sensitivity_y)); self.base.camera.setHpr(hpr.x-dx*cfg.orbit_sensitivity_x,pitch,0)
        else:
            self.orbit_yaw_deg=(self.orbit_yaw_deg+dx*cfg.orbit_sensitivity_x)%360.0; self.orbit_pitch_deg=max(cfg.orbit_pitch_min_deg,min(cfg.orbit_pitch_max_deg,self.orbit_pitch_deg-dy*cfg.orbit_sensitivity_y))
    def _update_mouse_look(self):
        if not self.mouse_look_active or self.base.win is None or not hasattr(self.base.win,"getPointer"): return
        cx=self.base.win.getXSize()//2; cy=self.base.win.getYSize()//2; pointer=self.base.win.getPointer(0); dx=pointer.getX()-cx; dy=pointer.getY()-cy
        if dx or dy: self._apply_mouse_delta(dx,dy); self._center_pointer()
    def _update_free_camera(self,dt:float):
        from panda3d.core import Vec3
        quat=self.base.camera.getQuat(self.base.render); movement=Vec3(0,0,0)
        movement+=quat.xform(Vec3(0,1,0))*((1 if self.key_state["forward"] else 0)-(1 if self.key_state["back"] else 0))
        movement+=quat.xform(Vec3(1,0,0))*((1 if self.key_state["right"] else 0)-(1 if self.key_state["left"] else 0))
        movement+=Vec3(0,0,1)*((1 if self.key_state["up"] else 0)-(1 if self.key_state["down"] else 0))
        if movement.lengthSquared()>0: movement.normalize(); multiplier=self.settings.camera.free_fast_multiplier if self.key_state["fast"] else 1.0; self.base.camera.setPos(self.base.camera.getPos()+movement*self.free_move_speed*multiplier*dt)
    def _toggle_path(self):
        if self.path_node: self.path_node.hide() if not self.path_node.isHidden() else self.path_node.show()
    def _step(self,direction:int):
        self.paused=True; self.elapsed=max(0,min(self.frames[-1]["time"],self.elapsed+direction*.1)); self.frame_index=max(0,min(len(self.frames)-1,int(round(self.elapsed/.1))-1))
        if direction<0: self.event_state.advance(self.elapsed)

    @staticmethod
    def _bar(value:float)->str:
        count=max(0,min(10,int(round(value*10)))); return "["+"#"*count+"-"*(10-count)+f"] {value*100:.0f}%"

    @staticmethod
    def _controls(values:dict[str,float])->str:
        return " ".join(f"{key.replace('throttle_','T')[0:3]}={value:+.2f}" for key,value in values.items())

    @staticmethod
    def _compact(text:str,limit:int=58)->str:
        clean=" ".join(text.replace("_"," ").split()); return clean if len(clean)<=limit else clean[:limit-1]+"…"

    def _update_overlays(self,time_s:float):
        from panda3d.core import ClockObject
        frame=self.replay_data.state_at(time_s); active=self.replay_data.active_failures(time_s); truth_ids=set(frame.ground_truth_ids)
        for failure_id,node in self.failure_markers.items(): node.show() if self.show_failures and failure_id in truth_ids else node.hide()
        events=self.event_state.advance(time_s)
        for event in events:
            if event.kind in {"activated","cleared"}:
                heading="MULTIPLE FAILURES ACTIVATED" if event.kind=="activated" and len(event.labels)>1 else "FAILURE ACTIVATED" if event.kind=="activated" else "FAILURE CLEARED"
                self.notification.setText(heading+"\n\n"+"\n".join(label.upper() for label in event.labels)); self.notification_frame.show(); self.notification_until=ClockObject.getGlobalClock().getRealTime()+3.6; self._log_event(f"failure_{event.kind}",event.time_s,failure_ids=event.failure_ids,labels=event.labels)
            elif event.monitor:
                self._log_event("monitor_response",event.time_s,status=event.monitor.status,reason=event.monitor.reason,violations=event.monitor.violated_constraints,fallback=event.monitor.fallback_activated,timeout=event.monitor.timeout,unknown=event.monitor.unknown,monitor_latency_ms=event.monitor.monitor_latency_ms)
                if event.monitor.status in {"modify","reject_fallback","unknown_fallback"} or event.monitor.timeout or event.monitor.unknown:
                    title="NEURAL ACTION MODIFIED" if event.monitor.status=="modify" else "SOLVER TIMEOUT\nFALLBACK CONTROLLER ACTIVE" if event.monitor.timeout else "SOLVER UNKNOWN\nFALLBACK CONTROLLER ACTIVE" if event.monitor.unknown else "NEURAL ACTION REJECTED\nFALLBACK CONTROLLER ACTIVE"
                    self.notification.setText(title+"\n\n"+self._compact("; ".join(event.monitor.violated_constraints) or event.monitor.reason)); self.notification_frame.show(); self.notification_until=ClockObject.getGlobalClock().getRealTime()+2.5
        if ClockObject.getGlobalClock().getRealTime()>self.notification_until: self.notification.setText(""); self.notification_frame.hide()
        if self.show_failures:
            names=lambda ids:[self._compact(next((f.display_name for f in self.replay_data.failures if f.failure_id==item),item),40) for item in ids]
            lines=["ACTIVE FAILURES"]+([f"• {self._compact(failure.display_name,40)}" for failure in active] or ["None"])
            if self.show_known: lines += ["","CONTROLLER-KNOWN"]+([f"• {name}" for name in names(frame.controller_known_ids)] or ["None"])
            if self.show_ground_truth: lines += ["","SIMULATOR GROUND TRUTH"]+([f"• {name}" for name in names(frame.ground_truth_ids)] or ["None"])
            self.failure_panel.setText("\n".join(lines)); self.failure_frame.show()
            for name,value in (("elevator",frame.elevator),("aileron",frame.aileron),("rudder",frame.rudder),("left_engine",frame.left_engine),("right_engine",frame.right_engine)):
                self.health_bars[name]["value"]=value*100; label={"elevator":"Elevator","aileron":"Aileron","rudder":"Rudder","left_engine":"Engine Left","right_engine":"Engine Right"}[name]; self.health_labels[name].setText(f"{label:<12} {value*100:3.0f}%")
        else: self.failure_panel.setText(""); self.failure_frame.hide()
        index=max(0,min(len(self.replay_data.monitor_events)-1,bisect_right([event.time_s for event in self.replay_data.monitor_events],time_s)-1)); monitor=self.replay_data.monitor_events[index]
        label="SOLVER TIMEOUT / FALLBACK" if monitor.timeout else "SOLVER UNKNOWN / FALLBACK" if monitor.unknown else {"accept":"ACCEPTED","modify":"MODIFIED","reject_fallback":"REJECTED / FALLBACK","unknown_fallback":"UNKNOWN / FALLBACK"}.get(monitor.status,monitor.status.upper())
        monitor_lines=["SAFETY MONITOR",f"Status: {label}",f"Latency: {monitor.monitor_latency_ms:.2f} ms"]
        if monitor.status!="accept" or monitor.timeout or monitor.unknown:
            monitor_lines += [f"Reason: {self._compact('; '.join(monitor.violated_constraints) or monitor.reason,48)}"]
        if self.show_monitor_details:
            monitor_lines += ["","Proposed:",f"  {self._controls(monitor.proposed)}","Executed:",f"  {self._controls(monitor.final)}",f"Constraints: {self._compact(', '.join(monitor.violated_constraints) or 'None',44)}"]
        else: monitor_lines += ["","M = action details"]
        self.monitor_panel.setText("\n".join(monitor_lines))
        past_fail=[event for event in self.replay_data.failure_events if event.time_s<=time_s]; past_monitor=[event for event in self.replay_data.monitor_events if event.time_s<=time_s and event.status!="accept"]; timeline=[(e.time_s,f"{', '.join(e.labels)} {e.kind}") for e in past_fail]+[(e.time_s,"Action modified" if e.status=="modify" else "Action rejected; fallback") for e in past_monitor]; timeline=sorted(timeline)[-5:]; self.timeline_panel.setText("RECENT EVENTS\n"+("\n".join(f"{time:5.2f}s  {self._compact(text,52)}" for time,text in timeline) or "No failure or monitor events yet"))

    def _interpolated(self,t):
        while self.frame_index+1<len(self.frames) and self.frames[self.frame_index+1]["time"]<t: self.frame_index+=1
        a=self.frames[self.frame_index]; b=self.frames[min(self.frame_index+1,len(self.frames)-1)]; span=max(b["time"]-a["time"],1e-9); u=max(0,min(1,(t-a["time"])/span)); pa=self.points[self.frame_index]; pb=self.points[min(self.frame_index+1,len(self.points)-1)]
        frame={k:a[k]+(b[k]-a[k])*u for k in ("pitch_deg","roll_deg","airspeed_kts","altitude_ft")}; frame["heading_deg"]=_short_heading_lerp(a["heading_deg"],b["heading_deg"],u); frame["time"]=t; return frame,tuple(pa[i]+(pb[i]-pa[i])*u for i in range(3))

    def _update(self,task):
        from panda3d.core import ClockObject,Vec3
        clock=ClockObject.getGlobalClock(); dt=clock.getDt()
        if self.auto_start_at is not None and clock.getRealTime()>=self.auto_start_at: self.paused=False; self.auto_start_at=None
        if not self.paused: self.elapsed+=dt*self.speed
        end=self.frames[-1]["time"]; self.elapsed=min(self.elapsed,end); frame,pos=self._interpolated(self.elapsed); self.aircraft_parent.setPos(*pos); self.aircraft_parent.setHpr(*replay_parent_hpr(frame["heading_deg"],frame["pitch_deg"],frame["roll_deg"]))
        self._update_mouse_look()
        forward=self.aircraft_parent.getQuat(self.base.render).xform(Vec3(0,1,0)); target=Vec3(*pos)+forward*self.settings.camera.look_ahead_m
        right=self.aircraft_parent.getQuat(self.base.render).xform(Vec3(1,0,0))
        alpha=1.0-math.exp(-self.settings.camera.smoothing*max(dt,0.0)); self.camera_distance+= (self.camera_distance_target-self.camera_distance)*alpha
        if self.camera_mode=="free": self._update_free_camera(dt)
        else:
            if self.camera_mode=="chase":
                yaw=math.radians(self.orbit_yaw_deg); pitch=math.radians(self.orbit_pitch_deg); orbit_forward=forward*math.cos(yaw)+right*math.sin(yaw); horizontal=self.camera_distance*math.cos(pitch); camera_pos=Vec3(*pos)-orbit_forward*horizontal+Vec3(0,0,self.settings.camera.chase_height_m+self.camera_distance*math.sin(pitch)); target=Vec3(*pos)+forward*self.settings.camera.look_ahead_m
            elif self.camera_mode=="side": camera_pos=Vec3(*pos)+right*20+Vec3(0,0,6); target=Vec3(*pos)+forward*5
            elif self.camera_mode=="top": camera_pos=Vec3(pos[0],pos[1],pos[2]+55); target=Vec3(*pos)
            elif self.camera_mode=="cockpit": camera_pos=Vec3(*pos)+forward*1.8+Vec3(0,0,1.35); target=Vec3(*pos)+forward*35+Vec3(0,0,1.0)
            else: camera_pos=Vec3(self.points[0][0]-45,self.points[0][1]-45,2.0); target=Vec3(*pos)
            camera_pos.z=max(camera_pos.z,2.0)
            if not self.camera_initialized: self.base.camera.setPos(camera_pos); self.smoothed_target=target; self.camera_initialized=True
            else: self.base.camera.setPos(self.base.camera.getPos()+(camera_pos-self.base.camera.getPos())*alpha); self.smoothed_target=self.smoothed_target+(target-self.smoothed_target)*alpha
            self.base.camera.lookAt(self.smoothed_target)
        if self.frame_index!=self.path_drawn_index: self._draw_path(self.frame_index,pos); self.path_drawn_index=self.frame_index
        self.telemetry.setText(f"t={frame['time']:.1f}s  speed={frame['airspeed_kts']:.1f} kt  heading={frame['heading_deg']:.1f}°  pitch={frame['pitch_deg']:.1f}°  roll={frame['roll_deg']:.1f}°\ntermination={self.summary.get('termination_reason')}  Space pause  R restart  Left/Right step  F failures  G truth  M monitor  T path  Esc exit")
        state="PAUSED" if self.paused else f"{self.speed:.1f}x"
        zoom=f"  {self.camera_distance_target:.1f} m" if self.camera_mode=="chase" else ""
        self.telemetry.setText(f"REPLAY  {state}\nTime       {frame['time']:6.1f} s\nAirspeed   {frame['airspeed_kts']:6.1f} kt\nAltitude   {frame['altitude_ft']:6.0f} ft\nHeading    {frame['heading_deg']:6.1f} deg\nPitch      {frame['pitch_deg']:+6.1f} deg\nRoll       {frame['roll_deg']:+6.1f} deg\nCamera     {self.camera_mode.upper()}{zoom}\n\nH = controls")
        self._update_overlays(frame["time"])
        return task.cont

    def run(self): self.base.run()
    def destroy(self): self.base.destroy()

def validate_model(settings:Replay3DSettings)->dict[str,Any]:
    """Load the configured model headlessly and report transformed dimensions/orientation."""
    from panda3d.core import Filename,NodePath,PandaNode,Vec3,loadPrcFileData
    loadPrcFileData("","window-type none")
    from direct.showbase.ShowBase import ShowBase
    base=ShowBase(windowType="none"); path=Path(settings.aircraft.model_path).resolve()
    try:
        model=base.loader.loadModel(Filename.fromOsSpecific(str(path)),noCache=True)
        if model.isEmpty(): raise RuntimeError("empty model")
        bounds=model.getTightBounds(); raw=tuple(float(bounds[1][i]-bounds[0][i]) for i in range(3)); dims={"wingspan_m":raw[0]*settings.aircraft.scale,"height_m":raw[1]*settings.aircraft.scale,"length_m":raw[2]*settings.aircraft.scale}
        transform=NodePath(PandaNode("offset")); transform.setHpr(settings.aircraft.yaw_offset_deg,settings.aircraft.pitch_offset_deg,settings.aircraft.roll_offset_deg); q=transform.getQuat(); forward=q.xform(Vec3(0,0,1)); up=q.xform(Vec3(0,1,0))
        return {"loaded":True,"model_path":str(path),"scale":settings.aircraft.scale,"offset_hpr_deg":[settings.aircraft.yaw_offset_deg,settings.aircraft.pitch_offset_deg,settings.aircraft.roll_offset_deg],"position_offset_m":list(settings.aircraft.position_offset_m),"dimensions_m":dims,"native_forward_after_offset":list(forward),"native_up_after_offset":list(up),"nose_points_forward":forward.y>0.99,"model_up_is_world_up":up.z>0.99,"textures_found":model.findAllTextures().getNumTextures()}
    except Exception as error: return {"loaded":False,"model_path":str(path),"fallback":"primitive","warning":f"{type(error).__name__}: {error}"}
    finally: base.destroy()
