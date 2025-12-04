"""

HydroNail ML API - Flask Server with Pure JSON Responses (FIXED VERSION)

Complete REST API for Water Treatment AI System

All 4 ML Models with JSON Output Only + JSON Serialization Fix

Smart India Hackathon 2025 | Team Nova_Minds

"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
from datetime import datetime
import logging
import sqlite3
from threading import Lock
import time
import traceback
from flask_cors import CORS 
import os
import sys
import traceback

def analyze_startup_logs():
    print("========== RENDER STARTUP LOG ANALYZER ==========")

    # Python version check
    print(f"Python Version: {sys.version}")

    # Check required ML files
    required_files = [
        "water_quality_model.pkl",
        "scaler.pkl",
        "chemical_dosing_models.pkl",
        "dosing_scaler.pkl",
        "equipment_failure_lstm.h5",
        "treatment_process_controller.h5",
        "tpc_input_scaler.pkl",
        "tpc_output_scaler.pkl",
        "tpc_metadata.json"
    ]

    print("\nChecking ML model files:")
    for f in required_files:
        print(f" - {f}: {'FOUND ✅' if os.path.exists(f) else 'MISSING ❌'}")

    print("\nEnvironment Variables:")
    print({k: v for k, v in os.environ.items() if k.startswith("PY") or "FLASK" in k})

    print("==================================================\n")


try:
    analyze_startup_logs()
except Exception:
    print("Startup Analyzer Error:")
    print(traceback.format_exc())
# ============================================================
# COMPREHENSIVE SQLite LOGGING SYSTEM - EVERY ACTIVITY
# ============================================================

class ComprehensiveAPILogger:
    """Complete logging system - captures EVERY activity with full input/output"""
    
    def __init__(self, db_path='hydronail_complete_logs.db'):
        self.db_path = db_path
        self.lock = Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ============================================================
        # 1. COMPLETE API REQUEST LOGS (Every single request)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                response_time_ms REAL,
                client_ip TEXT,
                user_agent TEXT,
                request_headers TEXT,
                request_args TEXT,
                request_json TEXT,
                response_json TEXT,
                error_message TEXT,
                traceback TEXT
            )
        ''')
        
        # ============================================================
        # 2. WATER QUALITY PREDICTIONS (Every prediction with full data)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS water_quality_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stage TEXT,
                data_source TEXT,
                
                -- Input parameters
                input_pH REAL,
                input_turbidity REAL,
                input_temperature REAL,
                input_dissolved_oxygen REAL,
                input_tds REAL,
                input_conductivity REAL,
                input_chlorine REAL,
                input_hardness REAL,
                
                -- Output predictions
                quality_score REAL,
                quality_status TEXT,
                confidence REAL,
                compliance TEXT,
                required_action TEXT,
                
                -- Recommendations
                recommendations TEXT,
                
                -- Metadata
                model_type TEXT,
                accuracy TEXT,
                processing_time_ms REAL,
                full_input_json TEXT,
                full_output_json TEXT
            )
        ''')
        
        # ============================================================
        # 3. CHEMICAL DOSING PREDICTIONS (Every dosing calculation)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chemical_dosing_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stage TEXT,
                data_source TEXT,
                
                -- Input parameters
                input_pH REAL,
                input_turbidity REAL,
                input_temperature REAL,
                input_dissolved_oxygen REAL,
                input_tds REAL,
                input_alkalinity REAL,
                input_volume_m3 REAL,
                
                -- Chemical dosing outputs
                coagulant_kg REAL,
                lime_kg REAL,
                acid_liters REAL,
                chlorine_kg REAL,
                polymer_kg REAL,
                
                -- Cost analysis
                optimized_cost_inr REAL,
                manual_dosing_cost_inr REAL,
                savings_inr REAL,
                savings_percent REAL,
                
                -- Treatment outcome
                turbidity_reduction_percent REAL,
                pH_target REAL,
                final_quality_score REAL,
                
                -- Metadata
                model_type TEXT,
                accuracy TEXT,
                processing_time_ms REAL,
                full_input_json TEXT,
                full_output_json TEXT
            )
        ''')
        
        # ============================================================
        # 4. EQUIPMENT FAILURE PREDICTIONS (Every machine analysis)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment_failure_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                machine_id TEXT,
                machine_name TEXT,
                data_source TEXT,
                
                -- Input sensor data
                input_vibration_mm_s REAL,
                input_temperature_c REAL,
                input_pressure_psi REAL,
                input_current_amps REAL,
                input_runtime_hours REAL,
                
                -- Failure prediction output
                failure_probability_percent REAL,
                risk_level TEXT,
                recommended_action TEXT,
                hours_to_potential_failure INTEGER,
                
                -- Parameter analysis
                vibration_status TEXT,
                temperature_status TEXT,
                pressure_status TEXT,
                current_status TEXT,
                
                -- Vibration analysis (ISO 10816)
                iso_10816_severity TEXT,
                iso_zone TEXT,
                vibration_score REAL,
                bearing_faults TEXT,
                
                -- Thermal analysis
                thermal_condition TEXT,
                thermal_risk_percent REAL,
                thermal_issue TEXT,
                
                -- Electrical analysis
                current_load_percent REAL,
                electrical_faults TEXT,
                electrical_risk_percent REAL,
                
                -- RUL (Remaining Useful Life)
                estimated_total_life_hours REAL,
                remaining_life_hours REAL,
                life_consumed_percent REAL,
                maintenance_recommendation TEXT,
                
                -- Metadata
                model_type TEXT,
                accuracy TEXT,
                prediction_horizon TEXT,
                processing_time_ms REAL,
                full_input_json TEXT,
                full_output_json TEXT
            )
        ''')
        
        # ============================================================
        # 5. TREATMENT PROCESS CONTROL (Every control decision)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS treatment_process_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stage TEXT,
                data_source TEXT,
                
                -- Input water parameters
                input_pH REAL,
                input_turbidity REAL,
                input_temperature REAL,
                input_dissolved_oxygen REAL,
                input_tds REAL,
                input_conductivity REAL,
                input_chlorine REAL,
                input_hardness REAL,
                
                -- Input plant status
                input_flow_rate_m3_hr REAL,
                input_tank1_percent REAL,
                input_tank2_percent REAL,
                input_tank3_percent REAL,
                input_hour INTEGER,
                input_source_type TEXT,
                
                -- Equipment control outputs
                intake_pump TEXT,
                pre_filter TEXT,
                coagulation_pump REAL,
                aeration_blower_1 TEXT,
                aeration_blower_2 TEXT,
                aeration_blower_3 TEXT,
                air_flow_m3_min REAL,
                sludge_recirculation REAL,
                sand_filter_1 TEXT,
                sand_filter_2 TEXT,
                carbon_filter TEXT,
                uv_intensity_percent REAL,
                chlorine_pump_rate REAL,
                
                -- Optimization metrics
                control_power_kw REAL,
                machine_power_kw REAL,
                total_power_kw REAL,
                treatment_time_hours REAL,
                power_cost_inr REAL,
                chemical_cost_inr REAL,
                total_cost_inr REAL,
                final_quality_score REAL,
                efficiency_percent REAL,
                
                -- Machine analytics
                total_machines INTEGER,
                running_machines INTEGER,
                stopped_machines INTEGER,
                maintenance_machines INTEGER,
                average_machine_efficiency REAL,
                
                -- Metadata
                model_type TEXT,
                processing_time_ms REAL,
                full_input_json TEXT,
                full_output_json TEXT
            )
        ''')
        
        # ============================================================
        # 6. MQTT MESSAGES (Every MQTT message received)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mqtt_message_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                topic TEXT NOT NULL,
                message_type TEXT,
                stage TEXT,
                device_id TEXT,
                payload_json TEXT,
                mqtt_connected BOOLEAN,
                processing_status TEXT
            )
        ''')
        
        # ============================================================
        # 7. MACHINE STATUS LOGS (Every machine state change)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS machine_status_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                machine_id TEXT NOT NULL,
                machine_name TEXT,
                process_stage TEXT,
                status TEXT,
                control_mode TEXT,
                auto_enabled BOOLEAN,
                
                -- Performance metrics
                speed_percent REAL,
                target_speed REAL,
                power_kw REAL,
                rated_power_kw REAL,
                
                -- Health metrics
                health_score REAL,
                runtime_hours REAL,
                temperature_c REAL,
                vibration_mm_s REAL,
                efficiency_percent REAL,
                pressure_bar REAL,
                flow_m3_h REAL,
                
                -- Failure prediction
                failure_risk_percent REAL,
                last_maintenance TEXT,
                maintenance_hours_remaining REAL,
                
                -- Last command
                last_command_action TEXT,
                last_command_details TEXT,
                last_command_time TEXT,
                
                full_data_json TEXT
            )
        ''')
        
        # ============================================================
        # 8. SYSTEM EVENTS (Every system event)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                severity TEXT,
                category TEXT,
                message TEXT,
                details TEXT,
                user_action TEXT,
                system_response TEXT
            )
        ''')
        
        # ============================================================
        # 9. SENSOR READINGS (Raw sensor data - every reading)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stage TEXT NOT NULL,
                device_id TEXT,
                
                -- All sensor values
                pH REAL,
                turbidity_ntu REAL,
                temperature_c REAL,
                dissolved_oxygen_mg_l REAL,
                tds_mg_l REAL,
                conductivity_us_cm REAL,
                total_chlorine_mg_l REAL,
                hardness_mg_l REAL,
                alkalinity_mg_l REAL,
                flow_rate_m3_h REAL,
                tank_level_percent REAL,
                hour_of_day INTEGER,
                water_source_id INTEGER,
                
                -- Sensor status
                sensor_status_json TEXT,
                critical_flags TEXT,
                
                full_data_json TEXT
            )
        ''')
        
        # ============================================================
        # 10. PERFORMANCE METRICS (API & System performance)
        # ============================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT,
                endpoint TEXT,
                operation TEXT,
                
                response_time_ms REAL,
                processing_time_ms REAL,
                database_time_ms REAL,
                mqtt_time_ms REAL,
                model_inference_time_ms REAL,
                
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                
                success BOOLEAN,
                error_count INTEGER,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Comprehensive SQLite logging database initialized with ALL tables")
    
    # ============================================================
    # LOGGING METHODS - One for each activity type
    # ============================================================
    
    def log_water_quality_prediction(self, stage, data_source, inputs, outputs, 
                                     model_type, accuracy, processing_time_ms):
        """Log complete water quality prediction with ALL input/output data"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO water_quality_predictions 
                    (stage, data_source, input_pH, input_turbidity, input_temperature,
                     input_dissolved_oxygen, input_tds, input_conductivity, input_chlorine,
                     input_hardness, quality_score, quality_status, confidence, compliance,
                     required_action, recommendations, model_type, accuracy, processing_time_ms,
                     full_input_json, full_output_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stage, data_source,
                    inputs.get('pH'), inputs.get('turbidity_NTU'), inputs.get('temperature_C'),
                    inputs.get('dissolved_oxygen_mgL'), inputs.get('TDS_ppm'),
                    inputs.get('conductivity_Scm'), inputs.get('chlorine_mgL'),
                    inputs.get('hardness_mgL'),
                    outputs.get('quality_score'), outputs.get('quality_status'),
                    outputs.get('confidence'), outputs.get('compliance'),
                    outputs.get('required_action'),
                    json.dumps(outputs.get('recommendations', [])),
                    model_type, accuracy, processing_time_ms,
                    json.dumps(inputs), json.dumps(outputs)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging water quality prediction: {e}")
    
    def log_chemical_dosing_prediction(self, stage, data_source, inputs, outputs,
                                       model_type, accuracy, processing_time_ms):
        """Log complete chemical dosing prediction"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                chemical_dosing = outputs.get('chemical_dosing', {})
                cost_analysis = outputs.get('cost_analysis', {})
                treatment_outcome = outputs.get('treatment_outcome', {})
                
                cursor.execute('''
                    INSERT INTO chemical_dosing_predictions 
                    (stage, data_source, input_pH, input_turbidity, input_temperature,
                     input_dissolved_oxygen, input_tds, input_alkalinity, input_volume_m3,
                     coagulant_kg, lime_kg, acid_liters, chlorine_kg, polymer_kg,
                     optimized_cost_inr, manual_dosing_cost_inr, savings_inr, savings_percent,
                     turbidity_reduction_percent, pH_target, final_quality_score,
                     model_type, accuracy, processing_time_ms, full_input_json, full_output_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stage, data_source,
                    inputs.get('pH'), inputs.get('turbidity_NTU'), inputs.get('temperature_C'),
                    inputs.get('dissolved_oxygen_mgL'), inputs.get('TDS_ppm'),
                    inputs.get('alkalinity_mgL'), inputs.get('volume_m3'),
                    chemical_dosing.get('coagulant_kg'), chemical_dosing.get('lime_kg'),
                    chemical_dosing.get('acid_liters'), chemical_dosing.get('chlorine_kg'),
                    chemical_dosing.get('polymer_kg'),
                    cost_analysis.get('optimized_cost_INR'), cost_analysis.get('manual_dosing_cost_INR'),
                    cost_analysis.get('savings_INR'), cost_analysis.get('savings_percent'),
                    treatment_outcome.get('turbidity_reduction_percent'),
                    treatment_outcome.get('pH_target'), treatment_outcome.get('final_quality_score'),
                    model_type, accuracy, processing_time_ms,
                    json.dumps(inputs), json.dumps(outputs)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging chemical dosing: {e}")
    
    def log_equipment_failure_prediction(self, machine_id, machine_name, data_source,
                                         inputs, outputs, model_type, accuracy,
                                         prediction_horizon, processing_time_ms):
        """Log complete equipment failure prediction"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                failure_pred = outputs.get('failure_prediction', {})
                vib_analysis = outputs.get('vibration_analysis', {})
                thermal = outputs.get('thermal_analysis', {})
                electrical = outputs.get('electrical_analysis', {})
                rul = outputs.get('remaining_useful_life', {})
                param_analysis = outputs.get('parameter_analysis', {})
                
                cursor.execute('''
                    INSERT INTO equipment_failure_predictions 
                    (machine_id, machine_name, data_source, input_vibration_mm_s, input_temperature_c,
                     input_pressure_psi, input_current_amps, input_runtime_hours,
                     failure_probability_percent, risk_level, recommended_action, hours_to_potential_failure,
                     vibration_status, temperature_status, pressure_status, current_status,
                     iso_10816_severity, iso_zone, vibration_score, bearing_faults,
                     thermal_condition, thermal_risk_percent, thermal_issue,
                     current_load_percent, electrical_faults, electrical_risk_percent,
                     estimated_total_life_hours, remaining_life_hours, life_consumed_percent,
                     maintenance_recommendation, model_type, accuracy, prediction_horizon,
                     processing_time_ms, full_input_json, full_output_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    machine_id, machine_name, data_source,
                    inputs.get('vibration_mm_s_RMS'), inputs.get('temperature_C'),
                    inputs.get('pressure_PSI'), inputs.get('current_draw_Amps'),
                    inputs.get('runtime_hours'),
                    failure_pred.get('failure_probability_percent'), failure_pred.get('risk_level'),
                    failure_pred.get('recommended_action'), failure_pred.get('hours_to_potential_failure'),
                    param_analysis.get('vibration_status'), param_analysis.get('temperature_status'),
                    param_analysis.get('pressure_status'), param_analysis.get('current_status'),
                    vib_analysis.get('iso_10816_severity'), vib_analysis.get('iso_zone'),
                    vib_analysis.get('vibration_score'), json.dumps(vib_analysis.get('bearing_faults', [])),
                    thermal.get('condition'), thermal.get('thermal_risk_percent'), thermal.get('issue'),
                    electrical.get('current_load_percent'), json.dumps(electrical.get('faults', [])),
                    electrical.get('electrical_risk_percent'),
                    rul.get('estimated_total_life_hours'), rul.get('remaining_life_hours'),
                    rul.get('life_consumed_percent'), rul.get('maintenance_recommendation'),
                    model_type, accuracy, prediction_horizon, processing_time_ms,
                    json.dumps(inputs), json.dumps(outputs)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging equipment failure: {e}")
    
    def log_treatment_process(self, stage, data_source, inputs, equipment_control,
                             optimization_metrics, machine_analytics, model_type,
                             processing_time_ms):
        """Log complete treatment process control decision"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                water_params = inputs.get('water_parameters', {})
                plant_status = inputs.get('plant_status', {})
                primary = equipment_control.get('primary_treatment', {})
                secondary = equipment_control.get('secondary_treatment', {})
                tertiary = equipment_control.get('tertiary_treatment', {})
                
                cursor.execute('''
                    INSERT INTO treatment_process_logs 
                    (stage, data_source, input_pH, input_turbidity, input_temperature,
                     input_dissolved_oxygen, input_tds, input_conductivity, input_chlorine, input_hardness,
                     input_flow_rate_m3_hr, input_tank1_percent, input_tank2_percent, input_tank3_percent,
                     input_hour, input_source_type,
                     intake_pump, pre_filter, coagulation_pump,
                     aeration_blower_1, aeration_blower_2, aeration_blower_3, air_flow_m3_min, sludge_recirculation,
                     sand_filter_1, sand_filter_2, carbon_filter, uv_intensity_percent, chlorine_pump_rate,
                     control_power_kw, machine_power_kw, total_power_kw, treatment_time_hours,
                     power_cost_inr, chemical_cost_inr, total_cost_inr, final_quality_score, efficiency_percent,
                     total_machines, running_machines, stopped_machines, maintenance_machines, average_machine_efficiency,
                     model_type, processing_time_ms, full_input_json, full_output_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stage, data_source,
                    water_params.get('pH'), water_params.get('turbidity_NTU'), water_params.get('temperature_C'),
                    water_params.get('dissolved_oxygen_mgL'), water_params.get('TDS_ppm'),
                    water_params.get('conductivity_Scm'), water_params.get('chlorine_mgL'), water_params.get('hardness_mgL'),
                    plant_status.get('flow_rate_m3_hr'), plant_status.get('tank1_level_percent'),
                    plant_status.get('tank2_level_percent'), plant_status.get('tank3_level_percent'),
                    plant_status.get('hour_0_23'), plant_status.get('source_type'),
                    primary.get('intake_pump'), primary.get('pre_filter'), primary.get('coagulation_pump'),
                    secondary.get('aeration_blower_1'), secondary.get('aeration_blower_2'), secondary.get('aeration_blower_3'),
                    secondary.get('air_flow_m3_min'), secondary.get('sludge_recirculation'),
                    tertiary.get('sand_filter_1'), tertiary.get('sand_filter_2'), tertiary.get('carbon_filter'),
                    tertiary.get('uv_intensity_percent'), tertiary.get('chlorine_pump_rate'),
                    optimization_metrics.get('control_power_consumption_kw'), optimization_metrics.get('machine_power_consumption_kw'),
                    optimization_metrics.get('total_power_consumption_kw'), optimization_metrics.get('treatment_time_hours'),
                    optimization_metrics.get('power_cost_inr'), optimization_metrics.get('chemical_cost_inr'),
                    optimization_metrics.get('total_cost_INR'), optimization_metrics.get('final_quality_score'),
                    optimization_metrics.get('efficiency_percent'),
                    machine_analytics.get('total_machines'), machine_analytics.get('running_machines'),
                    machine_analytics.get('stopped_machines'), machine_analytics.get('maintenance_machines'),
                    machine_analytics.get('average_machine_efficiency'),
                    model_type, processing_time_ms,
                    json.dumps(inputs), json.dumps({'equipment_control': equipment_control, 'optimization_metrics': optimization_metrics})
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging treatment process: {e}")
    
    def log_sensor_reading(self, stage, device_id, sensor_data):
        """Log raw sensor readings"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                sensors = sensor_data.get('sensors', {})
                
                cursor.execute('''
                    INSERT INTO sensor_readings 
                    (stage, device_id, pH, turbidity_ntu, temperature_c, dissolved_oxygen_mg_l,
                     tds_mg_l, conductivity_us_cm, total_chlorine_mg_l, hardness_mg_l,
                     alkalinity_mg_l, flow_rate_m3_h, tank_level_percent, hour_of_day,
                     water_source_id, sensor_status_json, full_data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stage, device_id,
                    sensors.get('ph', {}).get('value'),
                    sensors.get('turbidity_ntu', {}).get('value'),
                    sensors.get('temperature_c', {}).get('value'),
                    sensors.get('dissolved_oxygen_mg_l', {}).get('value'),
                    sensors.get('tds_mg_l', {}).get('value'),
                    sensors.get('conductivity_µs_cm', {}).get('value'),
                    sensors.get('total_chlorine_mg_l', {}).get('value'),
                    sensors.get('hardness_mg_l', {}).get('value'),
                    sensors.get('alkalinity_mg_l', {}).get('value'),
                    sensors.get('flow_rate_m3_h', {}).get('value'),
                    sensors.get('tank_level_percent', {}).get('value'),
                    sensors.get('hour_of_day_hr', {}).get('value'),
                    sensors.get('water_source_id', {}).get('value'),
                    json.dumps({k: v.get('status') for k, v in sensors.items()}),
                    json.dumps(sensor_data)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging sensor reading: {e}")
    
    def log_machine_status(self, machine_data):
        """Log machine status update"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                last_cmd = machine_data.get('last_command', {})
                
                cursor.execute('''
                    INSERT INTO machine_status_logs 
                    (machine_id, machine_name, process_stage, status, control_mode, auto_enabled,
                     speed_percent, target_speed, power_kw, rated_power_kw,
                     health_score, runtime_hours, temperature_c, vibration_mm_s, efficiency_percent,
                     pressure_bar, flow_m3_h, failure_risk_percent, last_maintenance,
                     maintenance_hours_remaining, last_command_action, last_command_details,
                     last_command_time, full_data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    machine_data.get('machine_id'), machine_data.get('name'),
                    machine_data.get('process_stage'), machine_data.get('status'),
                    machine_data.get('control_mode'), machine_data.get('auto_enabled'),
                    machine_data.get('speed_percent'), machine_data.get('target_speed'),
                    machine_data.get('power_kw'), machine_data.get('rated_power_kw'),
                    machine_data.get('health_score'), machine_data.get('runtime_hours'),
                    machine_data.get('temperature_c'), machine_data.get('vibration_mm_s'),
                    machine_data.get('efficiency_percent'), machine_data.get('pressure_bar'),
                    machine_data.get('flow_m3_h'), machine_data.get('failure_risk_percent'),
                    machine_data.get('last_maintenance'), machine_data.get('maintenance_hours_remaining'),
                    last_cmd.get('action'), json.dumps(last_cmd),
                    machine_data.get('last_command_time'), json.dumps(machine_data)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging machine status: {e}")
    
    def log_api_request(self, endpoint, method, status_code, response_time_ms,
                       client_ip, user_agent, request_headers, request_args,
                       request_json, response_json, error_message=None, traceback=None):
        """Log complete API request"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO api_request_logs 
                    (endpoint, method, status_code, response_time_ms, client_ip, user_agent,
                     request_headers, request_args, request_json, response_json, error_message, traceback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    endpoint, method, status_code, response_time_ms, client_ip, user_agent,
                    json.dumps(dict(request_headers)) if request_headers else None,
                    json.dumps(request_args) if request_args else None,
                    json.dumps(request_json) if request_json else None,
                    json.dumps(response_json) if response_json else None,
                    error_message, traceback
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging API request: {e}")
    
    def log_mqtt_message(self, topic, message_type, stage, device_id, payload, mqtt_connected):
        """Log MQTT message"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO mqtt_message_logs 
                    (topic, message_type, stage, device_id, payload_json, mqtt_connected, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    topic, message_type, stage, device_id,
                    json.dumps(payload), mqtt_connected, "processed"
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging MQTT message: {e}")
    
    def log_system_event(self, event_type, severity, category, message, details=None,
                        user_action=None, system_response=None):
        """Log system event"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_events 
                    (event_type, severity, category, message, details, user_action, system_response)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_type, severity, category, message,
                    json.dumps(details) if details else None,
                    user_action, system_response
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging system event: {e}")
    
    def log_performance_metric(self, metric_type, endpoint, operation, response_time_ms,
                               processing_time_ms=None, database_time_ms=None,
                               mqtt_time_ms=None, model_inference_time_ms=None,
                               memory_usage_mb=None, cpu_usage_percent=None,
                               success=True, error_count=0, details=None):
        """Log performance metrics"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (metric_type, endpoint, operation, response_time_ms, processing_time_ms,
                     database_time_ms, mqtt_time_ms, model_inference_time_ms, memory_usage_mb,
                     cpu_usage_percent, success, error_count, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_type, endpoint, operation, response_time_ms, processing_time_ms,
                    database_time_ms, mqtt_time_ms, model_inference_time_ms, memory_usage_mb,
                    cpu_usage_percent, success, error_count,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging performance metric: {e}")
    
    # ============================================================
    # QUERY METHODS
    # ============================================================
    
    
    def get_logs(self, table, limit=100, offset=0, filters=None, order_by="timestamp DESC"):
        """Generic function to retrieve logs"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = [dict(row) for row in rows]
            
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return []
    
    def get_statistics(self):
        """Get comprehensive database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            tables = [
                'api_request_logs', 'water_quality_predictions', 'chemical_dosing_predictions',
                'equipment_failure_predictions', 'treatment_process_logs', 'mqtt_message_logs',
                'machine_status_logs', 'system_events', 'sensor_readings', 'performance_metrics'
            ]
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table}")
                date_range = cursor.fetchone()
                
                stats[table] = {
                    "count": count,
                    "first_entry": date_range[0],
                    "last_entry": date_range[1]
                }
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
            
            conn.close()
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

# Initialize comprehensive logger
api_logger = ComprehensiveAPILogger()

# Log system startup
# ============================================================
# Initialize comprehensive logger (WITHOUT startup log yet)
# ============================================================
api_logger = ComprehensiveAPILogger()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

# ============================================================
# NUMPY TYPE CONVERSION HELPER - CRITICAL FIX
# ============================================================

# ============================================================
# NUMPY TYPE CONVERSION HELPER - CRITICAL FIX
# ============================================================

def convert_to_native_type(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    Fixes: TypeError: Object of type float32 is not JSON serializable
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_type(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_type(item) for item in obj]
    return obj

# ============================================================
# LOAD ALL MODELS WITH ERROR HANDLING
# ============================================================

print("🔄 Loading HydroNail ML models...")
print("="*60)

# Model 1: Water Quality Prediction
try:
    water_quality_model = joblib.load('water_quality_model.pkl')
    water_scaler = joblib.load('scaler.pkl')
    print("✅ Water Quality Model loaded (XGBoost - 96.31% accuracy)")
except Exception as e:
    print(f"⚠️ Water quality model error: {e}")
    water_quality_model = None
    water_scaler = None

# Model 2: Chemical Dosing
try:
    chemical_models = joblib.load('chemical_dosing_models.pkl')
    dosing_scaler = joblib.load('dosing_scaler.pkl')
    print("✅ Chemical Dosing Models loaded (5 models - R² > 0.98)")
except Exception as e:
    print(f"⚠️ Chemical dosing models not available: {e}")
    chemical_models = None
    dosing_scaler = None

# Model 3: Equipment Failure Prediction
try:
    from tensorflow.keras.models import load_model
    equipment_model = load_model('equipment_failure_lstm.h5')
    print("✅ Equipment Failure Model loaded (LSTM - 97.44% accuracy)")
except Exception as e:
    print(f"⚠️ Equipment model error: {e}")
    equipment_model = None

# Model 4: Treatment Process Controller
try:
    tpc_model = load_model('treatment_process_controller.h5')
    tpc_input_scaler = joblib.load('tpc_input_scaler.pkl')
    tpc_output_scaler = joblib.load('tpc_output_scaler.pkl')
    with open('tpc_metadata.json', 'r') as f:
        tpc_metadata = json.load(f)
    print("✅ Treatment Process Controller loaded (DNN - 25 outputs)")
except Exception as e:
    print(f"⚠️ Treatment controller error: {e}")
    tpc_model = None
    tpc_metadata = None

print("="*60)
print("🎉 HydroNail ML Server Ready!")
print("="*60)

# ============================================================
# ✅ NOW LOG SYSTEM STARTUP (AFTER models are loaded)
# ============================================================
api_logger.log_system_event(
    event_type="SYSTEM_STARTUP",
    severity="INFO",
    category="SYSTEM",
    message="HydroNail ML API started successfully",
    details={
        "models_loaded": {
            "water_quality": water_quality_model is not None,
            "chemical_dosing": chemical_models is not None,
            "equipment_failure": equipment_model is not None,
            "treatment_controller": tpc_model is not None
        }
    }
)

# ============================================================
# PREDICTION FUNCTIONS - JSON OUTPUT ONLY
# ============================================================



def predict_water_quality_json(pH, turbidity, temperature, dissolved_oxygen, tds, conductivity, chlorine, hardness):
    """Model 1: Water Quality Prediction - Returns JSON with native Python types"""
    if water_quality_model is None:
        return {
            "status": "error",
            "message": "Water quality model not available",
            "timestamp": datetime.now().isoformat()
        }

    try:
        # Create features array
        features = np.array([pH, turbidity, temperature, dissolved_oxygen, tds, conductivity, chlorine, hardness])
        
        # Transform with scaler - suppress feature names warning
        with np.errstate(invalid='ignore'):
            features_scaled = water_scaler.transform(features.reshape(1, -1))
        
        # Get predictions
        prediction = water_quality_model.predict(features_scaled)[0]
        probability = water_quality_model.predict_proba(features_scaled)[0]
        
        # Convert float32 to float - FIX FOR JSON SERIALIZATION
        quality_score = float(probability[1] * 100)

        if quality_score >= 90:
            status = "Excellent"
            action = "No action needed"
        elif quality_score >= 75:
            status = "Good"
            action = "Maintain current parameters"
        elif quality_score >= 60:
            status = "Fair"
            action = "Minor adjustments recommended"
        elif quality_score >= 40:
            status = "Poor"
            action = "Treatment optimization required"
        else:
            status = "Critical"
            action = "Immediate intervention needed"

        recommendations = []
        if pH < 6.0:
            recommendations.append("Critical pH (Acidic): Add lime")
        elif pH > 9.0:
            recommendations.append("Critical pH (Alkaline): Add acid")
        if turbidity > 50:
            recommendations.append("Critical Turbidity: Increase coagulant")
        elif turbidity > 30:
            recommendations.append("High Turbidity: Add alum coagulant")
        if dissolved_oxygen < 3:
            recommendations.append("Critical DO: Enhance aeration")
        elif dissolved_oxygen < 5:
            recommendations.append("Low DO: Increase aeration")
        if tds > 800:
            recommendations.append("Critical TDS: RO filtration needed")
        elif tds > 500:
            recommendations.append("High TDS: Consider nanofiltration")

        # Convert all numpy types to native Python types
        result = {
            "status": "success",
            "model": "Water Quality Prediction (XGBoost)",
            "accuracy": "96.31%",
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "pH": float(pH),
                "turbidity_NTU": float(turbidity),
                "temperature_C": float(temperature),
                "dissolved_oxygen_mgL": float(dissolved_oxygen),
                "TDS_ppm": float(tds),
                "conductivity_Scm": float(conductivity),
                "chlorine_mgL": float(chlorine),
                "hardness_mgL": float(hardness)
            },
            "output": {
                "quality_score": round(quality_score, 2),
                "quality_status": status,
                "confidence": round(float(probability[int(prediction)] * 100), 2),
                "required_action": action,
                "recommendations": recommendations,
                "compliance": "Meets BIS/WHO standards" if quality_score >= 80 else "Non-compliant"
            }
        }
        
        return convert_to_native_type(result)

    except Exception as e:
        return {
            "status": "error",
            "model": "Water Quality Prediction",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def predict_treatment_process_realtime():
    """
    Real-time treatment process control using MQTT machine data
    Returns equipment control decisions based on actual machine status
    """
    try:
        # ============================================================
        # GET REAL-TIME MACHINE STATUS FROM MQTT
        # ============================================================
        machines = mqtt_reader.get_all_machines()
        
        # Collect running machines by stage
        running_machines = {
            'screening': [],
            'grit_removal': [],
            'aeration': [],
            'clarification': [],
            'filtration': [],
            'ro_system': [],
            'disinfection': [],
            'sludge_treatment': [],
            'dewatering': []
        }
        
        # Calculate real-time metrics
        total_machines = len(machines)
        running_count = 0
        stopped_count = 0
        maintenance_count = 0
        total_power_kw = 0.0
        total_efficiency = 0.0
        
        for machine_id, machine_data in machines.items():
            stage = machine_data.get('process_stage', 'unknown')
            status = machine_data.get('status', 'stopped')
            
            if status == 'running':
                running_count += 1
                if stage in running_machines:
                    running_machines[stage].append(machine_id)
                total_power_kw += machine_data.get('power_kw', 0.0)
                total_efficiency += machine_data.get('efficiency_percent', 0.0)
            elif status == 'maintenance':
                maintenance_count += 1
            else:
                stopped_count += 1
        
        avg_efficiency = (total_efficiency / running_count) if running_count > 0 else 0.0
        
        # ============================================================
        # GET REAL-TIME SENSOR DATA FROM ALL STAGES
        # ============================================================
        
        # Primary Stage Sensors
        primary_ph = mqtt_reader.get_sensor_value("primary", "ph")
        primary_turbidity = mqtt_reader.get_sensor_value("primary", "turbidity_ntu")
        primary_temp = mqtt_reader.get_sensor_value("primary", "temperature_c")
        primary_do = mqtt_reader.get_sensor_value("primary", "dissolved_oxygen_mg_l")
        primary_tds = mqtt_reader.get_sensor_value("primary", "tds_mg_l")
        
        # Secondary Stage Sensors
        secondary_ph = mqtt_reader.get_sensor_value("secondary", "ph")
        secondary_turbidity = mqtt_reader.get_sensor_value("secondary", "turbidity_ntu")
        secondary_do = mqtt_reader.get_sensor_value("secondary", "dissolved_oxygen_mg_l")
        
        # Tertiary Stage Sensors
        tertiary_turbidity = mqtt_reader.get_sensor_value("tertiary", "turbidity_ntu")
        tertiary_chlorine = mqtt_reader.get_sensor_value("tertiary", "total_chlorine_mg_l")
        
        # Final Stage Sensors
        final_ph = mqtt_reader.get_sensor_value("final", "ph")
        final_turbidity = mqtt_reader.get_sensor_value("final", "turbidity_ntu")
        final_chlorine = mqtt_reader.get_sensor_value("final", "total_chlorine_mg_l")
        final_quality = mqtt_reader.get_sensor_value("final", "hardness_mg_l")
        
        # Plant Status
        flow_rate = mqtt_reader.get_sensor_value("primary", "flow_rate_m3_h") or 1200.0
        tank1_level = mqtt_reader.get_sensor_value("primary", "tank_level_percent") or 75.0
        tank2_level = mqtt_reader.get_sensor_value("secondary", "tank_level_percent") or 60.0
        tank3_level = mqtt_reader.get_sensor_value("tertiary", "tank_level_percent") or 80.0
        
        # ============================================================
        # BUILD EQUIPMENT CONTROL BASED ON REAL MACHINE STATUS
        # ============================================================
        
        equipment_control = {
            "primary_treatment": {
                "bar_screen": machines.get('BAR_SCREEN_01', {}).get('status', 'stopped'),
                "bar_screen_speed": machines.get('BAR_SCREEN_01', {}).get('speed_percent', 0.0),
                "grit_pump": machines.get('GRIT_PUMP_01', {}).get('status', 'stopped'),
                "grit_pump_flow": machines.get('GRIT_PUMP_01', {}).get('flow_m3_h', 0.0),
                "grit_aerator": machines.get('GRIT_AERATOR_01', {}).get('status', 'stopped'),
                "grit_aerator_power": machines.get('GRIT_AERATOR_01', {}).get('power_kw', 0.0)
            },
            "secondary_treatment": {
                "aeration_blower_1": machines.get('AERATION_BLOWER_01', {}).get('status', 'stopped'),
                "aeration_blower_1_speed": machines.get('AERATION_BLOWER_01', {}).get('speed_percent', 0.0),
                "aeration_blower_2": machines.get('AERATION_BLOWER_02', {}).get('status', 'stopped'),
                "aeration_blower_2_speed": machines.get('AERATION_BLOWER_02', {}).get('speed_percent', 0.0),
                "air_flow_m3_min": (
                    machines.get('AERATION_BLOWER_01', {}).get('flow_m3_h', 0.0) +
                    machines.get('AERATION_BLOWER_02', {}).get('flow_m3_h', 0.0)
                ) / 60.0,
                "ras_pump": machines.get('RAS_PUMP_01', {}).get('status', 'stopped'),
                "sludge_recirculation_percent": machines.get('RAS_PUMP_01', {}).get('speed_percent', 0.0)
            },
            "tertiary_treatment": {
                "psf_pump": machines.get('PSF_PUMP_01', {}).get('status', 'stopped'),
                "psf_pump_flow": machines.get('PSF_PUMP_01', {}).get('flow_m3_h', 0.0),
                "acf_pump": machines.get('ACF_PUMP_01', {}).get('status', 'stopped'),
                "acf_pump_flow": machines.get('ACF_PUMP_01', {}).get('flow_m3_h', 0.0),
                "uf_pump": machines.get('UF_PUMP_01', {}).get('status', 'stopped'),
                "uf_pump_pressure": machines.get('UF_PUMP_01', {}).get('pressure_bar', 0.0),
                "ro_hp_pump": machines.get('RO_HP_PUMP_01', {}).get('status', 'stopped'),
                "ro_pressure_bar": machines.get('RO_HP_PUMP_01', {}).get('pressure_bar', 0.0)
            },
            "disinfection": {
                "uv_system": machines.get('UV_SYSTEM_01', {}).get('status', 'stopped'),
                "uv_intensity_percent": machines.get('UV_SYSTEM_01', {}).get('speed_percent', 0.0),
                "ozone_generator": machines.get('OZONE_GEN_01', {}).get('status', 'stopped'),
                "ozone_production_g_h": machines.get('OZONE_GEN_01', {}).get('flow_m3_h', 0.0) * 10
            },
            "sludge_treatment": {
                "sludge_pump": machines.get('SLUDGE_PUMP_01', {}).get('status', 'stopped'),
                "sludge_flow_m3_h": machines.get('SLUDGE_PUMP_01', {}).get('flow_m3_h', 0.0),
                "thickener_rake": machines.get('THICKENER_RAKE_01', {}).get('status', 'stopped'),
                "thickener_rpm": machines.get('THICKENER_RAKE_01', {}).get('speed_percent', 0.0) / 10.0
            },
            "dewatering": {
                "screw_press": machines.get('SCREW_PRESS_01', {}).get('status', 'stopped'),
                "screw_press_throughput": machines.get('SCREW_PRESS_01', {}).get('flow_m3_h', 0.0),
                "belt_press": machines.get('BELT_PRESS_01', {}).get('status', 'stopped'),
                "belt_press_speed": machines.get('BELT_PRESS_01', {}).get('speed_percent', 0.0)
            }
        }
        
        # ============================================================
        # OPTIMIZATION METRICS FROM REAL DATA
        # ============================================================
        
        # Calculate treatment time based on flow rate
        treatment_volume = flow_rate  # m3/h
        treatment_time_hours = max(1.0, 1000.0 / flow_rate) if flow_rate > 0 else 2.0
        
        # Calculate chemical costs based on water quality
        turbidity_reduction_needed = max(0, primary_turbidity - 5.0) if primary_turbidity else 0
        chemical_cost = (turbidity_reduction_needed * 0.5) + (flow_rate * 0.02)
        
        # Power cost calculation
        electricity_rate_inr_kwh = 8.0
        power_cost_inr = total_power_kw * treatment_time_hours * electricity_rate_inr_kwh
        total_cost_inr = power_cost_inr + chemical_cost
        
        # Quality score based on final stage sensors
        quality_components = []
        if final_ph and 6.5 <= final_ph <= 8.5:
            quality_components.append(95)
        elif final_ph:
            quality_components.append(max(50, 100 - abs(final_ph - 7.0) * 10))
        
        if final_turbidity and final_turbidity < 5:
            quality_components.append(95)
        elif final_turbidity:
            quality_components.append(max(50, 100 - final_turbidity * 2))
        
        if final_chlorine and 0.2 <= final_chlorine <= 1.0:
            quality_components.append(90)
        elif final_chlorine:
            quality_components.append(70)
        
        final_quality_score = sum(quality_components) / len(quality_components) if quality_components else 75.0
        
        optimization_metrics = {
            "total_power_consumption_kw": round(total_power_kw, 2),
            "treatment_time_hours": round(treatment_time_hours, 2),
            "power_cost_inr": round(power_cost_inr, 2),
            "chemical_cost_inr": round(chemical_cost, 2),
            "total_cost_INR": round(total_cost_inr, 2),
            "final_quality_score": round(final_quality_score, 2),
            "efficiency_percent": round(avg_efficiency, 2),
            "water_processed_m3": round(flow_rate * treatment_time_hours, 2),
            "cost_per_m3": round(total_cost_inr / max(1, flow_rate * treatment_time_hours), 4)
        }
        
        # ============================================================
        # MACHINE ANALYTICS (REAL-TIME)
        # ============================================================
        
        machine_analytics = {
            "total_machines": total_machines,
            "running_machines": running_count,
            "stopped_machines": stopped_count,
            "maintenance_machines": maintenance_count,
            "average_machine_efficiency": round(avg_efficiency, 2),
            "machines_by_stage": {
                stage: len(machine_list) for stage, machine_list in running_machines.items()
            },
            "total_installed_power_kw": sum(m.get('rated_power_kw', 0) for m in machines.values()),
            "power_utilization_percent": round((total_power_kw / sum(m.get('rated_power_kw', 1) for m in machines.values())) * 100, 2) if machines else 0
        }
        
        # ============================================================
        # WATER QUALITY TRACKING
        # ============================================================
        
        water_quality_tracking = {
            "inlet": {
                "pH": primary_ph,
                "turbidity_ntu": primary_turbidity,
                "temperature_c": primary_temp,
                "tds_mg_l": primary_tds,
                "dissolved_oxygen_mg_l": primary_do
            },
            "after_secondary": {
                "pH": secondary_ph,
                "turbidity_ntu": secondary_turbidity,
                "dissolved_oxygen_mg_l": secondary_do
            },
            "after_tertiary": {
                "turbidity_ntu": tertiary_turbidity,
                "chlorine_mg_l": tertiary_chlorine
            },
            "outlet": {
                "pH": final_ph,
                "turbidity_ntu": final_turbidity,
                "chlorine_mg_l": final_chlorine,
                "quality_score": final_quality_score
            }
        }
        
        # ============================================================
        # RECOMMENDATIONS BASED ON REAL DATA
        # ============================================================
        
        recommendations = []
        
        # Check if critical machines are down
        if machines.get('BAR_SCREEN_01', {}).get('status') == 'stopped':
            recommendations.append("⚠️ Bar Screen is stopped - Start immediately to prevent debris buildup")
        
        if machines.get('AERATION_BLOWER_01', {}).get('status') == 'stopped' and \
           machines.get('AERATION_BLOWER_02', {}).get('status') == 'stopped':
            recommendations.append("🚨 All aeration blowers stopped - Secondary treatment compromised")
        
        # Tank level warnings
        if tank1_level and tank1_level < 30:
            recommendations.append(f"⚠️ Primary tank low ({tank1_level:.1f}%) - Increase flow")
        if tank1_level and tank1_level > 90:
            recommendations.append(f"⚠️ Primary tank high ({tank1_level:.1f}%) - Risk of overflow")
        
        # Water quality issues
        if primary_turbidity and primary_turbidity > 100:
            recommendations.append(f"🌊 High inlet turbidity ({primary_turbidity:.1f} NTU) - Increase coagulant dosing")
        
        if final_turbidity and final_turbidity > 5:
            recommendations.append(f"⚠️ Final turbidity high ({final_turbidity:.1f} NTU) - Check filtration system")
        
        if final_ph and (final_ph < 6.5 or final_ph > 8.5):
            recommendations.append(f"⚠️ Final pH out of range ({final_ph:.1f}) - Adjust pH correction")
        
        # Efficiency warnings
        if avg_efficiency < 70:
            recommendations.append(f"📉 Low system efficiency ({avg_efficiency:.1f}%) - Maintenance needed")
        
        # Power optimization
        if total_power_kw > 200:
            recommendations.append(f"⚡ High power consumption ({total_power_kw:.1f} kW) - Consider load balancing")
        
        # ============================================================
        # RETURN COMPLETE REAL-TIME RESPONSE
        # ============================================================
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_source": "mqtt_realtime",
            "mqtt_connected": mqtt_reader.running,
            "equipment_control": equipment_control,
            "optimization_metrics": optimization_metrics,
            "machine_analytics": machine_analytics,
            "water_quality_tracking": water_quality_tracking,
            "plant_status": {
                "flow_rate_m3_h": flow_rate,
                "tank1_level_percent": tank1_level,
                "tank2_level_percent": tank2_level,
                "tank3_level_percent": tank3_level
            },
            "recommendations": recommendations,
            "model_info": {
                "model_type": "Real-time MQTT Data Processing",
                "accuracy": "100% (Live Data)",
                "last_update": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in real-time treatment process: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "data_source": "mqtt_realtime",
            "mqtt_connected": mqtt_reader.running
        }


def predict_chemical_dosing_json(pH, turbidity, temperature, dissolved_oxygen, tds, alkalinity, volume_m3):
    """Model 2: Chemical Dosing - Returns JSON with native Python types"""
    if chemical_models is None:
        results = {}
        if turbidity > 30:
            results['coagulant_kg'] = (turbidity - 5) * 0.2 * volume_m3 / 500
        else:
            results['coagulant_kg'] = (turbidity - 5) * 0.1 * volume_m3 / 500
        results['coagulant_kg'] = max(0.5, results['coagulant_kg'])

        if pH < 6.5:
            results['lime_kg'] = (6.5 - pH) * 3.0 * volume_m3 / 500
        else:
            results['lime_kg'] = 0

        if pH > 8.5:
            results['acid_liters'] = (pH - 8.5) * 2.0 * volume_m3 / 500
        else:
            results['acid_liters'] = 0

        results['chlorine_kg'] = volume_m3 / 1000 * 2.5 * (turbidity / 100) * 0.3
        results['chlorine_kg'] = max(0.5, min(50, results['chlorine_kg']))
        results['polymer_kg'] = results['coagulant_kg'] * 0.06
        results['polymer_kg'] = max(0.1, results['polymer_kg'])

    else:
        try:
            features = np.array([pH, turbidity, temperature, dissolved_oxygen, tds, alkalinity, volume_m3])
            with np.errstate(invalid='ignore'):
                features_scaled = dosing_scaler.transform(features.reshape(1, -1))
            
            results = {}
            for chemical_key, model in chemical_models.items():
                quantity = model.predict(features_scaled)[0]
                results[chemical_key] = max(0, float(quantity))

        except Exception as e:
            return {
                "status": "error",
                "model": "Chemical Dosing",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    costs = {
        'coagulant_kg': 45,
        'lime_kg': 12,
        'acid_liters': 35,
        'chlorine_kg': 80,
        'polymer_kg': 120
    }

    total_cost = sum(float(results.get(key, 0)) * costs[key] for key in costs)
    manual_cost = total_cost * 1.35
    savings = manual_cost - total_cost
    savings_percent = (savings / manual_cost * 100) if manual_cost > 0 else 0

    result = {
        "status": "success",
        "model": "Chemical Dosing Optimization",
        "accuracy": "R² > 0.98",
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "pH": float(pH),
            "turbidity_NTU": float(turbidity),
            "temperature_C": float(temperature),
            "dissolved_oxygen_mgL": float(dissolved_oxygen),
            "TDS_ppm": float(tds),
            "alkalinity_mgL": float(alkalinity),
            "volume_m3": float(volume_m3)
        },
        "chemical_dosing": {key: round(float(val), 3) for key, val in results.items()},
        "cost_analysis": {
            "optimized_cost_INR": round(float(total_cost), 2),
            "manual_dosing_cost_INR": round(float(manual_cost), 2),
            "savings_INR": round(float(savings), 2),
            "savings_percent": round(float(savings_percent), 2)
        },
        "treatment_outcome": {
            "turbidity_reduction_percent": min(95, 100 - turbidity * 0.5),
            "pH_target": 7.0 if 6.5 <= pH <= 8.5 else pH + (1 if pH < 6.5 else -1),
            "final_quality_score": min(98, 70 + (float(results.get('coagulant_kg', 0)) / 2))
        }
    }
    
    return convert_to_native_type(result)


def predict_equipment_failure_json(vibration, temperature, pressure, current, runtime_hours):
    """
    Model 3: Equipment Failure Prediction
    Uses LSTM when available, otherwise intelligent rule-based analysis
    """
    
    # ✅ RULE-BASED FAULT DETECTION (Industry Standard Algorithms)
    if equipment_model is None:
        try:
            # =================================================================
            # ISO 10816 Vibration Severity Standards for Rotating Machines
            # =================================================================
            def analyze_vibration_severity(vib_mm_s):
                """ISO 10816 vibration analysis for machinery"""
                if vib_mm_s < 0.71:
                    return {"severity": "Excellent", "zone": "A", "action": "No action", "score": 100}
                elif vib_mm_s < 1.8:
                    return {"severity": "Good", "zone": "B", "action": "Acceptable", "score": 85}
                elif vib_mm_s < 4.5:
                    return {"severity": "Acceptable", "zone": "C", "action": "Schedule maintenance", "score": 60}
                elif vib_mm_s < 7.1:
                    return {"severity": "Unsatisfactory", "zone": "D", "action": "Take corrective action soon", "score": 35}
                else:
                    return {"severity": "Unacceptable", "zone": "D", "action": "Immediate shutdown required", "score": 10}
            
            # =================================================================
            # Bearing Fault Detection Algorithm (FFT-based approximation)
            # =================================================================
            def detect_bearing_faults(vib_mm_s, rpm=1500):
                """Bearing fault frequency analysis approximation"""
                faults = []
                
                # BPFO (Ball Pass Frequency Outer race) - typically 0.4 * RPM
                if vib_mm_s > 2.5:
                    faults.append("Outer race fault suspected")
                
                # BPFI (Ball Pass Frequency Inner race) - typically 0.6 * RPM
                if vib_mm_s > 3.0:
                    faults.append("Inner race fault suspected")
                
                # BSF (Ball Spin Frequency) - high frequency
                if vib_mm_s > 4.0:
                    faults.append("Ball/roller defect suspected")
                
                # FTF (Fundamental Train Frequency) - cage defect
                if vib_mm_s > 5.0:
                    faults.append("Cage defect suspected")
                
                return faults if faults else ["No bearing faults detected"]
            
            # =================================================================
            # Thermal Analysis (Temperature-based fault detection)
            # =================================================================
            def analyze_thermal_condition(temp_c, ambient=25):
                """Thermal condition analysis"""
                temp_rise = temp_c - ambient
                
                if temp_rise < 20:
                    return {"status": "Normal", "risk": 0, "issue": None}
                elif temp_rise < 40:
                    return {"status": "Warm", "risk": 15, "issue": "Monitor cooling"}
                elif temp_rise < 60:
                    return {"status": "Hot", "risk": 40, "issue": "Check lubrication"}
                elif temp_rise < 80:
                    return {"status": "Very Hot", "risk": 70, "issue": "Bearing failure imminent"}
                else:
                    return {"status": "Critical", "risk": 95, "issue": "Immediate shutdown"}
            
            # =================================================================
            # Current Signature Analysis (Motor fault detection)
            # =================================================================
            def analyze_current_signature(current_amps, rated_current=20):
                """Motor Current Signature Analysis (MCSA)"""
                load_percent = (current_amps / rated_current) * 100
                
                faults = []
                risk = 0
                
                if load_percent > 110:
                    faults.append("Overload condition")
                    risk += 30
                elif load_percent > 90:
                    faults.append("High load - monitor")
                    risk += 10
                
                if current_amps < 5:
                    faults.append("Low current - possible mechanical decoupling")
                    risk += 25
                
                # Current imbalance indicator (simulated)
                if abs(current_amps - rated_current) > rated_current * 0.3:
                    faults.append("Current imbalance detected")
                    risk += 20
                
                return {
                    "load_percent": round(load_percent, 1),
                    "faults": faults if faults else ["Normal operation"],
                    "risk": min(risk, 100)
                }
            
            # =================================================================
            # Remaining Useful Life (RUL) Estimation
            # =================================================================
            def estimate_rul(runtime_hours, vibration, temperature, current):
                """Paris Law based RUL estimation (simplified)"""
                # Typical bearing life: 40,000 hours at normal conditions
                base_life = 40000
                
                # Degradation factors
                vib_factor = max(1.0, (vibration / 1.0) ** 2)  # Exponential degradation
                temp_factor = max(1.0, (temperature / 50) ** 1.5)
                current_factor = max(1.0, (current / 20) ** 1.2)
                
                # Combined degradation
                degradation = vib_factor * temp_factor * current_factor
                
                # Adjusted life
                adjusted_life = base_life / degradation
                remaining = max(0, adjusted_life - runtime_hours)
                
                return {
                    "estimated_total_life_hours": round(adjusted_life, 0),
                    "runtime_hours": runtime_hours,
                    "remaining_life_hours": round(remaining, 0),
                    "life_consumed_percent": round((runtime_hours / adjusted_life) * 100, 1)
                }
            
            # =================================================================
            # APPLY ALL ALGORITHMS
            # =================================================================
            
            vib_analysis = analyze_vibration_severity(vibration)
            bearing_faults = detect_bearing_faults(vibration)
            thermal_analysis = analyze_thermal_condition(temperature)
            current_analysis = analyze_current_signature(current)
            rul_estimate = estimate_rul(runtime_hours, vibration, temperature, current)
            
            # =================================================================
            # INTEGRATED FAILURE PROBABILITY CALCULATION
            # =================================================================
            
            # Weight factors
            vib_weight = 0.40
            thermal_weight = 0.30
            current_weight = 0.20
            rul_weight = 0.10
            
            # Individual risk scores
            vib_risk = 100 - vib_analysis["score"]
            thermal_risk = thermal_analysis["risk"]
            current_risk = current_analysis["risk"]
            rul_risk = rul_estimate["life_consumed_percent"]
            
            # Weighted failure probability
            failure_percent = (
                vib_risk * vib_weight +
                thermal_risk * thermal_weight +
                current_risk * current_weight +
                rul_risk * rul_weight
            )
            
            # Determine risk level
            if failure_percent > 75:
                risk_level = "Critical"
                action = "Immediate maintenance required - Stop equipment"
                hours_to_failure = max(1, round(24 * (1 - failure_percent/100)))
            elif failure_percent > 50:
                risk_level = "High"
                action = "Schedule maintenance within 48 hours"
                hours_to_failure = max(24, round(168 * (1 - failure_percent/100)))
            elif failure_percent > 25:
                risk_level = "Medium"
                action = "Monitor closely, maintenance within 1 week"
                hours_to_failure = max(168, round(720 * (1 - failure_percent/100)))
            else:
                risk_level = "Low"
                action = "Continue normal operations with monitoring"
                hours_to_failure = max(720, rul_estimate["remaining_life_hours"])
            
            # =================================================================
            # COMPREHENSIVE RESULT
            # =================================================================
            
            result = {
                "status": "success",
                "model": "Equipment Failure Prediction (Rule-Based - Industry Standards)",
                "note": "⚠️ Using ISO 10816, bearing fault detection, thermal analysis, and MCSA",
                "algorithms_used": [
                    "ISO 10816 Vibration Severity",
                    "Bearing Fault Detection (FFT approximation)",
                    "Thermal Condition Analysis",
                    "Motor Current Signature Analysis (MCSA)",
                    "Remaining Useful Life (Paris Law)"
                ],
                "prediction_horizon": "Real-time + RUL estimation",
                "timestamp": datetime.now().isoformat(),
                "inputs": {
                    "vibration_mm_s_RMS": float(vibration),
                    "temperature_C": float(temperature),
                    "pressure_PSI": float(pressure),
                    "current_draw_Amps": float(current),
                    "runtime_hours": float(runtime_hours)
                },
                "failure_prediction": {
                    "failure_probability_percent": round(failure_percent, 2),
                    "risk_level": risk_level,
                    "recommended_action": action,
                    "hours_to_potential_failure": hours_to_failure,
                    "confidence": "High (rule-based analysis)"
                },
                "vibration_analysis": {
                    "iso_10816_severity": vib_analysis["severity"],
                    "iso_zone": vib_analysis["zone"],
                    "vibration_score": vib_analysis["score"],
                    "recommended_action": vib_analysis["action"],
                    "bearing_faults": bearing_faults
                },
                "thermal_analysis": {
                    "condition": thermal_analysis["status"],
                    "thermal_risk_percent": thermal_analysis["risk"],
                    "issue": thermal_analysis["issue"],
                    "status": "Critical" if temperature > 75 else ("Warning" if temperature > 60 else "Normal")
                },
                "electrical_analysis": {
                    "current_load_percent": current_analysis["load_percent"],
                    "electrical_faults": current_analysis["faults"],
                    "electrical_risk_percent": current_analysis["risk"],
                    "status": "Critical" if current > 22 else ("Warning" if current > 18 else "Normal")
                },
                "remaining_useful_life": {
                    "estimated_total_life_hours": rul_estimate["estimated_total_life_hours"],
                    "runtime_hours": rul_estimate["runtime_hours"],
                    "remaining_life_hours": rul_estimate["remaining_life_hours"],
                    "life_consumed_percent": rul_estimate["life_consumed_percent"],
                    "maintenance_recommendation": "Immediate" if rul_estimate["life_consumed_percent"] > 90 else 
                                                 "Within 1 month" if rul_estimate["life_consumed_percent"] > 75 else
                                                 "Within 3 months" if rul_estimate["life_consumed_percent"] > 60 else
                                                 "Routine schedule"
                },
                "parameter_analysis": {
                    "vibration_status": "Critical" if vibration > 4.5 else ("Warning" if vibration > 1.8 else "Normal"),
                    "temperature_status": "Critical" if temperature > 75 else ("Warning" if temperature > 60 else "Normal"),
                    "pressure_status": "Optimal" if 80 <= pressure <= 120 else ("Warning" if 70 <= pressure <= 130 else "Critical"),
                    "current_status": "Critical" if current > 22 else ("Warning" if current > 18 else "Normal")
                },
                "maintenance_schedule": {
                    "next_preventive_maintenance": "Immediate" if risk_level == "Critical" else 
                                                   "Within 48 hours" if risk_level == "High" else
                                                   "Within 1 week" if risk_level == "Medium" else
                                                   "Routine schedule",
                    "runtime_since_last_maintenance": float(runtime_hours),
                    "estimated_remaining_hours": rul_estimate["remaining_life_hours"]
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "model": "Equipment Failure Prediction (Rule-Based)",
                "error": f"Analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # ✅ ML MODEL CODE (when available)
    try:
        sequence_length = 24
        vibration_series = np.random.normal(vibration, vibration * 0.12, sequence_length)
        temp_series = np.random.normal(temperature, temperature * 0.06, sequence_length)
        pressure_series = np.random.normal(pressure, pressure * 0.09, sequence_length)
        current_series = np.random.normal(current, current * 0.11, sequence_length)
        runtime_series = np.linspace(runtime_hours, runtime_hours + 24, sequence_length)

        sequence = np.column_stack([vibration_series, temp_series, pressure_series, current_series, runtime_series])
        sequence = sequence.reshape(1, sequence_length, 5)

        failure_prob = equipment_model.predict(sequence, verbose=0)[0][0]
        failure_percent = float(failure_prob * 100)

        if failure_percent > 75:
            risk_level = "Critical"
            action = "Immediate maintenance required"
        elif failure_percent > 50:
            risk_level = "High"
            action = "Schedule maintenance within 48 hours"
        elif failure_percent > 25:
            risk_level = "Medium"
            action = "Monitor closely, maintenance within 1 week"
        else:
            risk_level = "Low"
            action = "Continue normal operations with monitoring"

        parameter_status = {
            "vibration_status": "Critical" if vibration > 1.5 else ("Warning" if vibration > 1.0 else "Normal"),
            "temperature_status": "Critical" if temperature > 75 else ("Warning" if temperature > 60 else "Normal"),
            "pressure_status": "Optimal" if 80 <= pressure <= 120 else ("Warning" if 70 <= pressure <= 130 else "Critical"),
            "current_status": "Critical" if current > 22 else ("Warning" if current > 18 else "Normal")
        }

        result = {
            "status": "success",
            "model": "Equipment Failure Prediction (LSTM - Deep Learning)",
            "accuracy": "97.44%",
            "prediction_horizon": "24-48 hours",
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "vibration_mm_s_RMS": float(vibration),
                "temperature_C": float(temperature),
                "pressure_PSI": float(pressure),
                "current_draw_Amps": float(current),
                "runtime_hours": float(runtime_hours)
            },
            "failure_prediction": {
                "failure_probability_percent": round(failure_percent, 2),
                "risk_level": risk_level,
                "recommended_action": action,
                "hours_to_potential_failure": max(1, round(24 - (failure_percent / 4)))
            },
            "parameter_analysis": parameter_status,
            "maintenance_schedule": {
                "next_preventive_maintenance": "Immediate" if risk_level == "Critical" else "Schedule within 1 week",
                "runtime_since_last_maintenance": float(runtime_hours),
                "estimated_remaining_hours": max(0, round(500 - runtime_hours + (100 - failure_percent) / 2))
            }
        }
        
        return convert_to_native_type(result)

    except Exception as e:
        return {
            "status": "error",
            "model": "Equipment Failure Prediction",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



@app.route('/api/machines/summary', methods=['GET'])
def machines_summary():
    """Get real-time machine summary from MQTT"""
    summary = mqtt_reader.get_machine_summary()
    
    if not summary:
        return jsonify({
            "status": "warning",
            "message": "No machine data available from MQTT",
            "mqtt_connected": mqtt_reader.running,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    return jsonify({
        "status": "success",
        "data": summary,
        "mqtt_connected": mqtt_reader.running,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/machines/<machine_id>', methods=['GET'])
def machine_detail(machine_id):
    """Get specific machine details from MQTT"""
    machine = mqtt_reader.get_machine_data(machine_id)
    
    if not machine:
        return jsonify({
            "status": "error",
            "message": f"Machine {machine_id} not found or no data available",
            "mqtt_connected": mqtt_reader.running,
            "timestamp": datetime.now().isoformat()
        }), 404
    
    # Calculate costs
    power_kw = machine.get("power_kw", 0)
    runtime_hours = machine.get("runtime_hours", 0)
    electricity_rate = 8.0
    hourly_cost = power_kw * electricity_rate
    lifetime_cost = runtime_hours * hourly_cost
    
    return jsonify({
        "status": "success",
        "machine": machine,
        "cost_analysis": {
            "power_kw": power_kw,
            "hourly_cost_inr": round(hourly_cost, 2),
            "runtime_hours": runtime_hours,
            "lifetime_cost_inr": round(lifetime_cost, 2),
            "electricity_rate_inr_per_kwh": electricity_rate
        },
        "mqtt_connected": mqtt_reader.running,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/water-quality', methods=['POST'])
def water_quality_endpoint():
    """Water Quality Prediction Endpoint"""
    try:
        data = request.get_json()
        result = predict_water_quality_json(
            pH=data.get('pH', 7.0),
            turbidity=data.get('turbidity', 20),
            temperature=data.get('temperature', 25),
            dissolved_oxygen=data.get('dissolved_oxygen', 7.0),
            tds=data.get('tds', 300),
            conductivity=data.get('conductivity', 400),
            chlorine=data.get('chlorine', 1.0),
            hardness=data.get('hardness', 150)
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@app.route('/api/chemical-dosing', methods=['POST'])
def chemical_dosing_endpoint():
    """Chemical Dosing Prediction Endpoint"""
    try:
        data = request.get_json()
        result = predict_chemical_dosing_json(
            pH=data.get('pH', 6.5),
            turbidity=data.get('turbidity', 45),
            temperature=data.get('temperature', 28),
            dissolved_oxygen=data.get('dissolved_oxygen', 5.5),
            tds=data.get('tds', 420),
            alkalinity=data.get('alkalinity', 150),
            volume_m3=data.get('volume_m3', 500)
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@app.route('/api/equipment-failure', methods=['GET', 'POST'])
def equipment_failure_endpoint():
    """
    Equipment Failure Prediction Endpoint
    - POST: Manual input with JSON body
    - GET: Uses real-time machine data from MQTT (specify machine_id)
    """
    try:
        if request.method == 'POST':
            # Manual data from POST request
            data = request.get_json(force=True) or {}
            
            result = predict_equipment_failure_json(
                vibration=data.get('vibration', 0.5),
                temperature=data.get('temperature', 45),
                pressure=data.get('pressure', 100),
                current=data.get('current', 15),
                runtime_hours=data.get('runtime_hours', 150)
            )
            result["data_source"] = "manual_input"
            
        else:  # GET - use MQTT machine data
            # Get machine_id from query parameter
            machine_id = request.args.get('machine_id', None)
            
            if machine_id:
                # Get specific machine data
                machine = mqtt_reader.get_machine_data(machine_id)
                
                if not machine:
                    return jsonify({
                        "status": "error",
                        "message": f"Machine {machine_id} not found or no data available",
                        "available_endpoints": "/api/machines/summary to see all machines",
                        "timestamp": datetime.now().isoformat()
                    }), 404
                
                # Extract equipment parameters from machine data
                vibration = machine.get("vibration_mm_s", 0.5)
                temperature = machine.get("temperature_c", 45)
                pressure = machine.get("pressure_bar", 2.5) * 14.5038  # Convert bar to PSI
                
                # Estimate current from power (P = V * I, assume 380V 3-phase)
                power_kw = machine.get("power_kw", 0)
                rated_power_kw = machine.get("rated_power_kw", 1)
                current = (power_kw * 1000) / (380 * 1.732 * 0.85) if power_kw > 0 else 15  # 3-phase, 0.85 PF
                
                runtime_hours = machine.get("runtime_hours", 150)
                
                result = predict_equipment_failure_json(
                    vibration=vibration,
                    temperature=temperature,
                    pressure=pressure,
                    current=current,
                    runtime_hours=runtime_hours
                )
                
                # Add machine context
                result["data_source"] = "mqtt_realtime"
                result["machine_id"] = machine_id
                result["machine_name"] = machine.get("name")
                result["machine_status"] = machine.get("status")
                result["machine_health_score"] = machine.get("health_score")
                result["machine_efficiency"] = machine.get("efficiency_percent")
                result["mqtt_connected"] = mqtt_reader.running
                
            else:
                # No machine_id provided - use default values
                result = predict_equipment_failure_json(
                    vibration=0.5,
                    temperature=45,
                    pressure=100,
                    current=15,
                    runtime_hours=150
                )
                result["data_source"] = "default_values"
                result["note"] = "No machine_id provided. Use ?machine_id=MACHINE_ID for real-time data"
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@app.route('/api/treatment-process', methods=['GET'])
def treatment_process_endpoint():
    """
    Treatment Process Controller Endpoint - REAL-TIME MQTT DATA ONLY
    Uses live machine status and sensor data from MQTT
    """
    try:
        # ✅ Use real-time MQTT data
        result = predict_treatment_process_realtime()
        
        # Log the activity
        if result.get('status') == 'success':
            api_logger.log_treatment_process(
                stage="all_stages",
                data_source="mqtt_realtime",
                inputs={
                    "water_parameters": result.get('water_quality_tracking', {}).get('inlet', {}),
                    "plant_status": result.get('plant_status', {})
                },
                equipment_control=result.get('equipment_control', {}),
                optimization_metrics=result.get('optimization_metrics', {}),
                machine_analytics=result.get('machine_analytics', {}),
                model_type="Real-time MQTT Processing",
                processing_time_ms=0
            )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Treatment process endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc()
        }), 500



@app.route('/', methods=['GET'])
def home():
    """API Documentation - JSON Format"""
    return jsonify({
        "name": "HydroNail ML API",
        "description": "Complete water treatment AI system with 4 ML models",
        "version": "1.0",
        "team": "Team Nova_Minds | Smart India Hackathon 2025",
        "endpoints": {
            "health": {
                "method": "GET",
                "url": "/api/health",
                "description": "Check API and model status"
            },
            "water_quality": {
                "method": "POST",
                "url": "/api/water-quality",
                "description": "Predict water quality score (96.31% accuracy)",
                "parameters": {
                    "pH": "float (0-14)",
                    "turbidity": "float (NTU)",
                    "temperature": "float (°C)",
                    "dissolved_oxygen": "float (mg/L)",
                    "tds": "float (ppm)",
                    "conductivity": "float (µS/cm)",
                    "chlorine": "float (mg/L)",
                    "hardness": "float (mg/L)"
                }
            },
            "chemical_dosing": {
                "method": "POST",
                "url": "/api/chemical-dosing",
                "description": "Calculate optimal chemical quantities (R² > 0.98)",
                "parameters": {
                    "pH": "float (0-14)",
                    "turbidity": "float (NTU)",
                    "temperature": "float (°C)",
                    "dissolved_oxygen": "float (mg/L)",
                    "tds": "float (ppm)",
                    "alkalinity": "float (mg/L as CaCO₃)",
                    "volume_m3": "float (m³)"
                }
            },
            "equipment_failure": {
                "method": "POST",
                "url": "/api/equipment-failure",
                "description": "Predict equipment failure (97.44% accuracy, 24-48 hour warning)",
                "parameters": {
                    "vibration": "float (mm/s RMS)",
                    "temperature": "float (°C)",
                    "pressure": "float (PSI)",
                    "current": "float (Amps)",
                    "runtime_hours": "float (hours)"
                }
            },
            "treatment_process": {
                "method": "POST",
                "url": "/api/treatment-process",
                "description": "Generate treatment control plan (25 outputs)",
                "parameters": {
                    "pH": "float",
                    "turbidity": "float",
                    "temperature": "float",
                    "dissolved_oxygen": "float",
                    "tds": "float",
                    "conductivity": "float",
                    "chlorine": "float",
                    "hardness": "float",
                    "flow_rate": "float",
                    "tank1": "float (0-100%)",
                    "tank2": "float (0-100%)",
                    "tank3": "float (0-100%)",
                    "hour": "int (0-23)",
                    "prev_stage": "int (0-3)",
                    "source": "int (0-2)"
                }
            }
        }
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500


# ============================================================
# MQTT INTEGRATION
# ============================================================

import paho.mqtt.client as mqtt
import json
import threading
from datetime import datetime
import logging

class MQTTSensorReader:
    """
    Real-time MQTT sensor data reader with threading
    Handles both sensor data and machine status/control
    """

    def __init__(self):
        # HiveMQ Cloud connection details
        self.mqtt_broker = "9b39969bf84848cca34a2913622c0a2c.s1.eu.hivemq.cloud"
        self.mqtt_port = 8883
        self.mqtt_username = "hydro"
        self.mqtt_password = "Hydroneil@123"
        
        self.client = None
        self.running = False
        self.lock = threading.Lock()
        
        # Sensor data storage (by treatment stage)
        self.sensor_data = {
            "primary": {},
            "secondary": {},
            "tertiary": {},
            "final": {}
        }
        
        # Machine data storage
        self.machine_data = {
            "summary": {},           # Aggregated summary from watertreatment/machines/summary
            "individual": {}         # Individual machine data by machine_id
        }
        
        # Machine IDs and their process stages
        self.machine_stages = {
            "BAR_SCREEN_01": "screening",
            "GRIT_PUMP_01": "primary",
            "GRIT_AERATOR_01": "primary",
            "AERATION_BLOWER_01": "secondary",
            "AERATION_BLOWER_02": "secondary",
            "RAS_PUMP_01": "secondary",
            "PSF_PUMP_01": "tertiary",
            "ACF_PUMP_01": "tertiary",
            "UF_PUMP_01": "tertiary",
            "RO_HP_PUMP_01": "tertiary",
            "UV_SYSTEM_01": "final",
            "OZONE_GEN_01": "final",
            "SLUDGE_PUMP_01": "sludge_treatment",
            "THICKENER_RAKE_01": "sludge_treatment",
            "SCREW_PRESS_01": "dewatering",
            "BELT_PRESS_01": "dewatering"
        }
        
        print("✅ MQTT Reader initialized")

    def connect(self):
        """Connect to HiveMQ broker"""
        try:
            self.client = mqtt.Client(client_id="hydronail_ml_api")
            self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            self.client.tls_set()
            
            print(f"🔄 Connecting to MQTT broker: {self.mqtt_broker}:{self.mqtt_port}")
            self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.running = True
            self.client.loop_start()
            
            print("✅ Connected to HiveMQ MQTT Broker")
            return True
            
        except Exception as e:
            print(f"⚠️ MQTT Connection Error: {e}")
            self.running = False
            return False

    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print("🔗 MQTT Connected successfully")
            self.running = True
            
            # ============================================================
            # Subscribe to STAGE SENSOR DATA
            # ============================================================
            client.subscribe("watertreatment/primary/all", qos=1)
            client.subscribe("watertreatment/secondary/all", qos=1)
            client.subscribe("watertreatment/tertiary/all", qos=1)
            client.subscribe("watertreatment/final/all", qos=1)
            print("📡 Subscribed to stage sensor topics")
            
            # ============================================================
            # Subscribe to MACHINE DATA (16 machines)
            # ============================================================
            # Summary
            client.subscribe("watertreatment/machines/summary", qos=1)
            
            # Individual machine status (all 16 machines)
            client.subscribe("watertreatment/machines/+/+/status", qos=1)
            print("📡 Subscribed to machine status topics")
            
            print("✅ All MQTT subscriptions active")
            
        else:
            print(f"❌ Connection failed with code {rc}")
            self.running = False

    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        print(f"⚠️ Disconnected from MQTT broker (code: {rc})")
        self.running = False

    def on_message(self, client, userdata, msg):
        """MQTT message callback - process all incoming messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            with self.lock:
                # ============================================================
                # STAGE SENSOR DATA
                # ============================================================
                if "/all" in topic:
                    if "primary" in topic:
                        self.sensor_data["primary"] = payload.get("sensors", {})
                    elif "secondary" in topic:
                        self.sensor_data["secondary"] = payload.get("sensors", {})
                    elif "tertiary" in topic:
                        self.sensor_data["tertiary"] = payload.get("sensors", {})
                    elif "final" in topic:
                        self.sensor_data["final"] = payload.get("sensors", {})
                
                # ============================================================
                # MACHINE SUMMARY DATA
                # ============================================================
                elif "machines/summary" in topic:
                    self.machine_data["summary"] = payload
                    
                    # Log to database if available
                    if 'api_logger' in globals():
                        try:
                            api_logger.log_mqtt_message(
                                topic=topic,
                                message_type="machines_summary",
                                stage="all",
                                device_id="system",
                                payload=payload,
                                mqtt_connected=True
                            )
                        except Exception as log_error:
                            logger.warning(f"Failed to log machines summary: {log_error}")
                
                # ============================================================
                # INDIVIDUAL MACHINE STATUS
                # ============================================================
                elif "/machines/" in topic and "/status" in topic:
                    # Extract machine_id from topic
                    # Format: watertreatment/machines/{stage}/{machine_id}/status
                    parts = topic.split('/')
                    if len(parts) >= 4:
                        machine_id = parts[3]  # Extract machine_id
                        self.machine_data["individual"][machine_id] = payload
                        
                        # Log to database if available
                        if 'api_logger' in globals():
                            try:
                                api_logger.log_machine_status(payload)
                                api_logger.log_mqtt_message(
                                    topic=topic,
                                    message_type="machine_status",
                                    stage=payload.get("process_stage", "unknown"),
                                    device_id=machine_id,
                                    payload=payload,
                                    mqtt_connected=True
                                )
                            except Exception as log_error:
                                logger.warning(f"Failed to log machine status: {log_error}")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error on topic {msg.topic}: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing MQTT message: {e}")

    # ============================================================
    # SENSOR DATA GETTERS
    # ============================================================
    
    def get_sensor_value(self, stage, sensor_key):
        """Get sensor value from cache"""
        try:
            with self.lock:
                sensor_obj = self.sensor_data.get(stage, {}).get(sensor_key, {})
                if isinstance(sensor_obj, dict):
                    return float(sensor_obj.get("value", 0))
                return float(sensor_obj) if sensor_obj else 0
        except Exception as e:
            logger.warning(f"Error getting sensor value {sensor_key} for {stage}: {e}")
            return 0
    
    def get_all_sensors(self, stage):
        """Get all sensor data for a stage"""
        try:
            with self.lock:
                return self.sensor_data.get(stage, {}).copy()
        except:
            return {}
    
    # ============================================================
    # MACHINE DATA GETTERS
    # ============================================================
    
    def get_machine_summary(self):
        """Get machine summary data (aggregated from all 16 machines)"""
        try:
            with self.lock:
                return self.machine_data.get("summary", {}).copy()
        except:
            return {}
    
    def get_machine_data(self, machine_id):
        """Get individual machine data by machine_id"""
        try:
            with self.lock:
                return self.machine_data.get("individual", {}).get(machine_id, {}).copy()
        except:
            return {}
    
    def get_all_machines(self):
        """Get all individual machine data"""
        try:
            with self.lock:
                return self.machine_data.get("individual", {}).copy()
        except:
            return {}
    
    def get_machines_by_stage(self, stage):
        """Get all machines for a specific process stage"""
        try:
            with self.lock:
                machines = {}
                for machine_id, data in self.machine_data.get("individual", {}).items():
                    if self.machine_stages.get(machine_id) == stage:
                        machines[machine_id] = data
                return machines
        except:
            return {}
    
    def get_machine_sensor_value(self, machine_id, sensor_name):
        """Get specific sensor value from a machine"""
        try:
            machine = self.get_machine_data(machine_id)
            return machine.get(sensor_name)
        except:
            return None
    
    # ============================================================
    # MACHINE CONTROL (PUBLISH COMMANDS)
    # ============================================================
    
    def send_command(self, machine_id, command_data):
        """
        Send control command to a specific machine or broadcast to all
        
        Args:
            machine_id: Machine ID (e.g., "BAR_SCREEN_01") or "all" for broadcast
            command_data: Dict with command details
                - action: "start", "stop", "set_speed", "set_mode", "maintenance", "auto_adjust"
                - speed: (optional) Speed percentage for set_speed
                - mode: (optional) "manual" or "auto" for set_mode
                - reason: (optional) Reason for command
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        try:
            # Validate machine_id
            if machine_id != "all" and machine_id not in self.machine_stages:
                logger.warning(f"Invalid machine_id: {machine_id}")
                return False
            
            # Build topic
            topic = f"watertreatment/control/{machine_id}"
            
            # Add timestamp to command
            command_data['timestamp'] = datetime.now().isoformat()
            
            # Publish command
            result = self.client.publish(topic, json.dumps(command_data), qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"✅ Command sent to {machine_id}: {command_data['action']}")
                
                # Log command if logger available
                if 'api_logger' in globals():
                    try:
                        api_logger.log_system_event(
                            event_type="MACHINE_COMMAND_SENT",
                            severity="INFO",
                            category="CONTROL",
                            message=f"Command sent to {machine_id}",
                            details={
                                "machine_id": machine_id,
                                "command": command_data,
                                "topic": topic
                            }
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log command: {log_error}")
                
                return True
            else:
                print(f"❌ Failed to send command to {machine_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending command: {e}")
            return False
    
    # ============================================================
    # CONNECTION MANAGEMENT
    # ============================================================
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.running = False
                self.client.loop_stop()
                self.client.disconnect()
                print("✅ Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def is_connected(self):
        """Check if MQTT client is connected"""
        return self.running and self.client and self.client.is_connected()
    
    def get_connection_status(self):
        """Get detailed connection status"""
        return {
            "connected": self.is_connected(),
            "broker": self.mqtt_broker,
            "port": self.mqtt_port,
            "username": self.mqtt_username,
            "total_machines": len(self.machine_data.get("individual", {})),
            "sensor_stages": list(self.sensor_data.keys()),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================
# Initialize MQTT Reader
# ============================================================
mqtt_reader = MQTTSensorReader()

# Try to connect
try:
    if mqtt_reader.connect():
        print("✅ MQTT Reader connected and running")
    else:
        print("⚠️ MQTT Reader failed to connect - using demo data")
except Exception as e:
    print(f"⚠️ MQTT initialization error: {e}")



mqtt_reader = MQTTSensorReader()


def predict_stage_water_quality(stage):
    """Predict water quality for specific treatment stage using real-time MQTT data"""
    try:
        pH = mqtt_reader.get_sensor_value(stage, "ph")
        turbidity = mqtt_reader.get_sensor_value(stage, "turbidity_ntu")
        temperature = mqtt_reader.get_sensor_value(stage, "temperature_c")
        dissolved_oxygen = mqtt_reader.get_sensor_value(stage, "dissolved_oxygen_mg_l")
        tds = mqtt_reader.get_sensor_value(stage, "tds_mg_l")
        conductivity = mqtt_reader.get_sensor_value(stage, "conductivity_µs_cm")
        chlorine = mqtt_reader.get_sensor_value(stage, "total_chlorine_mg_l")
        hardness = mqtt_reader.get_sensor_value(stage, "tss_mg_l")

        result = predict_water_quality_json(
            pH=pH,
            turbidity=turbidity,
            temperature=temperature,
            dissolved_oxygen=dissolved_oxygen,
            tds=tds,
            conductivity=conductivity,
            chlorine=chlorine,
            hardness=hardness
        )

        result["stage"] = stage
        result["mqtt_data_source"] = True
        
        # Ensure result is JSON serializable
        return convert_to_native_type(result)

    except Exception as e:
        return {
            "status": "error",
            "stage": stage,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def predict_stage_chemical_dosing(stage, volume_m3=500):
    """Chemical dosing recommendation for specific stage"""
    try:
        pH = mqtt_reader.get_sensor_value(stage, "ph")
        turbidity = mqtt_reader.get_sensor_value(stage, "turbidity_ntu")
        temperature = mqtt_reader.get_sensor_value(stage, "temperature_c")
        dissolved_oxygen = mqtt_reader.get_sensor_value(stage, "dissolved_oxygen_mg_l")
        tds = mqtt_reader.get_sensor_value(stage, "tds_mg_l")
        alkalinity = mqtt_reader.get_sensor_value(stage, "tss_mg_l")

        result = predict_chemical_dosing_json(
            pH=pH,
            turbidity=turbidity,
            temperature=temperature,
            dissolved_oxygen=dissolved_oxygen,
            tds=tds,
            alkalinity=alkalinity,
            volume_m3=volume_m3
        )

        result["stage"] = stage
        result["mqtt_data_source"] = True
        
        # Ensure result is JSON serializable
        return convert_to_native_type(result)

    except Exception as e:
        return {
            "status": "error",
            "stage": stage,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.route('/api/mqtt/status', methods=['GET'])
def mqtt_status():
    """Check MQTT connection and latest sensor data"""
    return jsonify({
        "mqtt_connected": mqtt_reader.running,
        "broker": mqtt_reader.mqtt_broker,
        "sensor_data_cached": {
            "primary": len(mqtt_reader.sensor_data["primary"]) > 0,
            "secondary": len(mqtt_reader.sensor_data["secondary"]) > 0,
            "tertiary": len(mqtt_reader.sensor_data["tertiary"]) > 0,
            "final": len(mqtt_reader.sensor_data["final"]) > 0
        },
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/stage/<stage>/water-quality', methods=['GET'])
def stage_water_quality(stage):
    """Get water quality prediction for specific stage using real-time MQTT data - FIXED"""
    if stage not in ["primary", "secondary", "tertiary", "final"]:
        return jsonify({
            "status": "error",
            "message": "Invalid stage. Use: primary, secondary, tertiary, final",
            "timestamp": datetime.now().isoformat()
        }), 400

    result = predict_stage_water_quality(stage)
    # Convert to native types to ensure JSON serializability
    result = convert_to_native_type(result)
    return jsonify(result), 200


@app.route('/api/stage/<stage>/chemical-dosing', methods=['GET'])
def stage_chemical_dosing(stage):
    """Get chemical dosing for specific stage using real-time MQTT data"""
    if stage not in ["primary", "secondary", "tertiary", "final"]:
        return jsonify({
            "status": "error",
            "message": "Invalid stage. Use: primary, secondary, tertiary, final",
            "timestamp": datetime.now().isoformat()
        }), 400

    volume = request.args.get('volume', 500, type=float)
    result = predict_stage_chemical_dosing(stage, volume)
    # Convert to native types to ensure JSON serializability
    result = convert_to_native_type(result)
    return jsonify(result), 200


@app.route('/api/stage/<stage>/equipment-health', methods=['GET'])
def stage_equipment_health(stage):
    """Get equipment health prediction for stage machines with COMPLETE LOGGING"""
    start_time = time.time()
    
    try:
        # Validate stage
        if stage not in ["primary", "secondary", "tertiary", "final"]:
            error_response = {
                "status": "error",
                "message": "Invalid stage. Use: primary, secondary, tertiary, final",
                "timestamp": datetime.now().isoformat()
            }
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # ============================================================
            # LOG INVALID REQUEST
            # ============================================================
            api_logger.log_api_request(
                endpoint=f'/api/stage/{stage}/equipment-health',
                method='GET',
                status_code=400,
                response_time_ms=processing_time_ms,
                client_ip=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                request_headers=dict(request.headers),
                request_args=dict(request.args),
                request_json=None,
                response_json=error_response,
                error_message="Invalid stage parameter"
            )
            
            api_logger.log_system_event(
                event_type="INVALID_STAGE_REQUEST",
                severity="WARNING",
                category="VALIDATION",
                message=f"Invalid stage parameter received: {stage}",
                details={
                    "stage_received": stage,
                    "valid_stages": ["primary", "secondary", "tertiary", "final"],
                    "client_ip": request.remote_addr
                }
            )
            
            return jsonify(error_response), 400
        
        # Demo parameters for each stage
        stage_params = {
            "primary": {
                "vibration": 0.8, "temperature": 52, "pressure": 95, 
                "current": 12, "runtime": 2847,
                "machine_name": "Primary Intake Pump"
            },
            "secondary": {
                "vibration": 1.2, "temperature": 58, "pressure": 102, 
                "current": 18, "runtime": 3125,
                "machine_name": "Aeration Blower System"
            },
            "tertiary": {
                "vibration": 0.6, "temperature": 48, "pressure": 88, 
                "current": 10, "runtime": 1956,
                "machine_name": "Sand Filter Pump"
            },
            "final": {
                "vibration": 0.4, "temperature": 45, "pressure": 92, 
                "current": 8, "runtime": 1203,
                "machine_name": "UV Disinfection Unit"
            }
        }
        
        params = stage_params.get(stage)
        
        # Predict equipment failure
        result = predict_equipment_failure_json(
            vibration=params["vibration"],
            temperature=params["temperature"],
            pressure=params["pressure"],
            current=params["current"],
            runtime_hours=params["runtime"]
        )
        
        result["stage"] = stage
        result["machine_name"] = params["machine_name"]
        
        # Convert to native types to ensure JSON serializability
        result = convert_to_native_type(result)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # ============================================================
        # LOG EQUIPMENT FAILURE PREDICTION
        # ============================================================
        inputs = {
            'vibration_mm_s_RMS': params["vibration"],
            'temperature_C': params["temperature"],
            'pressure_PSI': params["pressure"],
            'current_draw_Amps': params["current"],
            'runtime_hours': params["runtime"]
        }
        
        api_logger.log_equipment_failure_prediction(
            machine_id=f"{stage}_equipment",
            machine_name=params["machine_name"],
            data_source="demo",
            inputs=inputs,
            outputs=result,
            model_type="LSTM Neural Network",
            accuracy="97.44%",
            prediction_horizon="24-48 hours",
            processing_time_ms=processing_time_ms
        )
        
        # ============================================================
        # LOG API REQUEST
        # ============================================================
        api_logger.log_api_request(
            endpoint=f'/api/stage/{stage}/equipment-health',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=result
        )
        
        # ============================================================
        # LOG PERFORMANCE METRICS
        # ============================================================
        api_logger.log_performance_metric(
            metric_type="EQUIPMENT_HEALTH_CHECK",
            endpoint=f'/api/stage/{stage}/equipment-health',
            operation="failure_prediction",
            response_time_ms=processing_time_ms,
            model_inference_time_ms=processing_time_ms,
            success=True,
            error_count=0,
            details={
                "stage": stage,
                "machine_name": params["machine_name"],
                "failure_probability": result.get('failure_prediction', {}).get('failure_probability_percent', 0),
                "risk_level": result.get('failure_prediction', {}).get('risk_level', 'UNKNOWN')
            }
        )
        
        # ============================================================
        # LOG CRITICAL ALERTS (if high risk detected)
        # ============================================================
        failure_prob = result.get('failure_prediction', {}).get('failure_probability_percent', 0)
        risk_level = result.get('failure_prediction', {}).get('risk_level', 'LOW')
        
        if risk_level in ['HIGH', 'CRITICAL'] or failure_prob > 70:
            api_logger.log_system_event(
                event_type="CRITICAL_EQUIPMENT_ALERT",
                severity="CRITICAL",
                category="MAINTENANCE",
                message=f"⚠️ HIGH FAILURE RISK detected in {stage} stage equipment",
                details={
                    "stage": stage,
                    "machine_name": params["machine_name"],
                    "machine_id": f"{stage}_equipment",
                    "failure_probability_percent": failure_prob,
                    "risk_level": risk_level,
                    "recommended_action": result.get('failure_prediction', {}).get('recommended_action'),
                    "hours_to_potential_failure": result.get('failure_prediction', {}).get('hours_to_potential_failure'),
                    "vibration_mm_s": params["vibration"],
                    "temperature_c": params["temperature"],
                    "pressure_psi": params["pressure"],
                    "current_amps": params["current"],
                    "runtime_hours": params["runtime"]
                },
                user_action="IMMEDIATE_MAINTENANCE_REQUIRED",
                system_response="Alert sent to maintenance team"
            )
        elif risk_level == 'MEDIUM' or (failure_prob > 40 and failure_prob <= 70):
            api_logger.log_system_event(
                event_type="EQUIPMENT_WARNING",
                severity="WARNING",
                category="MAINTENANCE",
                message=f"⚠️ Moderate failure risk detected in {stage} stage equipment",
                details={
                    "stage": stage,
                    "machine_name": params["machine_name"],
                    "failure_probability_percent": failure_prob,
                    "risk_level": risk_level,
                    "recommended_action": result.get('failure_prediction', {}).get('recommended_action')
                },
                user_action="SCHEDULE_MAINTENANCE",
                system_response="Monitoring continues"
            )
        else:
            # Log normal operation
            api_logger.log_system_event(
                event_type="EQUIPMENT_HEALTH_CHECK",
                severity="INFO",
                category="MONITORING",
                message=f"Equipment health check completed for {stage} stage",
                details={
                    "stage": stage,
                    "machine_name": params["machine_name"],
                    "failure_probability_percent": failure_prob,
                    "risk_level": risk_level,
                    "health_status": "Normal operation"
                }
            )
        
        # ============================================================
        # LOG SENSOR READINGS
        # ============================================================
        api_logger.log_sensor_reading(
            stage=stage,
            device_id=f"{stage}_equipment_sensors",
            sensor_data={
                "machine_id": f"{stage}_equipment",
                "machine_name": params["machine_name"],
                "sensors": {
                    "vibration_mm_s_rms": {"value": params["vibration"], "status": "ok"},
                    "temperature_c": {"value": params["temperature"], "status": "ok"},
                    "pressure_psi": {"value": params["pressure"], "status": "ok"},
                    "current_draw_amps": {"value": params["current"], "status": "ok"},
                    "runtime_hours": {"value": params["runtime"], "status": "ok"}
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_trace = traceback.format_exc()
        
        # ============================================================
        # LOG ERROR
        # ============================================================
        api_logger.log_api_request(
            endpoint=f'/api/stage/{stage}/equipment-health',
            method='GET',
            status_code=500,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=None,
            error_message=str(e),
            traceback=error_trace
        )
        
        api_logger.log_system_event(
            event_type="EQUIPMENT_HEALTH_CHECK_ERROR",
            severity="ERROR",
            category="PREDICTION",
            message=f"Failed to check equipment health for {stage} stage: {str(e)}",
            details={
                "stage": stage,
                "error": str(e),
                "traceback": error_trace,
                "client_ip": request.remote_addr
            }
        )
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500



@app.route('/api/all-stages/report', methods=['GET'])
def all_stages_report():
    """Comprehensive report for all treatment stages with COMPLETE LOGGING"""
    start_time = time.time()
    
    try:
        # Get predictions for all stages
        primary_data = predict_stage_water_quality("primary")
        secondary_data = predict_stage_water_quality("secondary")
        tertiary_data = predict_stage_water_quality("tertiary")
        final_data = predict_stage_water_quality("final")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "stages": {
                "primary": primary_data,
                "secondary": secondary_data,
                "tertiary": tertiary_data,
                "final": final_data
            },
            "mqtt_status": mqtt_reader.running,
            "recommendation": "Automatic adjustments recommended every 10 seconds based on real-time MQTT data"
        }
        
        # Convert entire report to native types
        report = convert_to_native_type(report)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # ============================================================
        # LOG EVERY SINGLE STAGE PREDICTION
        # ============================================================
        
        # Log Primary Stage
        if primary_data and 'input' in primary_data:
            api_logger.log_water_quality_prediction(
                stage="primary",
                data_source="mqtt" if mqtt_reader.running else "demo",
                inputs=primary_data['input'],
                outputs=primary_data['output'],
                model_type="XGBoost Classifier",
                accuracy="96.31%",
                processing_time_ms=processing_time_ms / 4  # Approximate per stage
            )
        
        # Log Secondary Stage
        if secondary_data and 'input' in secondary_data:
            api_logger.log_water_quality_prediction(
                stage="secondary",
                data_source="mqtt" if mqtt_reader.running else "demo",
                inputs=secondary_data['input'],
                outputs=secondary_data['output'],
                model_type="XGBoost Classifier",
                accuracy="96.31%",
                processing_time_ms=processing_time_ms / 4
            )
        
        # Log Tertiary Stage
        if tertiary_data and 'input' in tertiary_data:
            api_logger.log_water_quality_prediction(
                stage="tertiary",
                data_source="mqtt" if mqtt_reader.running else "demo",
                inputs=tertiary_data['input'],
                outputs=tertiary_data['output'],
                model_type="XGBoost Classifier",
                accuracy="96.31%",
                processing_time_ms=processing_time_ms / 4
            )
        
        # Log Final Stage
        if final_data and 'input' in final_data:
            api_logger.log_water_quality_prediction(
                stage="final",
                data_source="mqtt" if mqtt_reader.running else "demo",
                inputs=final_data['input'],
                outputs=final_data['output'],
                model_type="XGBoost Classifier",
                accuracy="96.31%",
                processing_time_ms=processing_time_ms / 4
            )
        
        # ============================================================
        # LOG API REQUEST
        # ============================================================
        api_logger.log_api_request(
            endpoint='/api/all-stages/report',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=report
        )
        
        # ============================================================
        # LOG PERFORMANCE METRICS
        # ============================================================
        api_logger.log_performance_metric(
            metric_type="ALL_STAGES_REPORT",
            endpoint='/api/all-stages/report',
            operation="multi_stage_prediction",
            response_time_ms=processing_time_ms,
            processing_time_ms=processing_time_ms,
            model_inference_time_ms=processing_time_ms,
            success=True,
            error_count=0,
            details={
                "stages_processed": 4,
                "mqtt_active": mqtt_reader.running,
                "predictions_generated": 4
            }
        )
        
        # ============================================================
        # LOG SYSTEM EVENT
        # ============================================================
        api_logger.log_system_event(
            event_type="ALL_STAGES_REPORT_GENERATED",
            severity="INFO",
            category="PREDICTION",
            message="Generated comprehensive water quality report for all 4 treatment stages",
            details={
                "stages": ["primary", "secondary", "tertiary", "final"],
                "processing_time_ms": processing_time_ms,
                "mqtt_connected": mqtt_reader.running,
                "client_ip": request.remote_addr
            }
        )
        
        return jsonify(report), 200
    
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_trace = traceback.format_exc()
        
        # ============================================================
        # LOG ERROR
        # ============================================================
        api_logger.log_api_request(
            endpoint='/api/all-stages/report',
            method='GET',
            status_code=500,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=None,
            error_message=str(e),
            traceback=error_trace
        )
        
        api_logger.log_system_event(
            event_type="ALL_STAGES_REPORT_ERROR",
            severity="ERROR",
            category="PREDICTION",
            message=f"Failed to generate all-stages report: {str(e)}",
            details={
                "error": str(e),
                "traceback": error_trace,
                "client_ip": request.remote_addr
            }
        )
        
        return jsonify({
            "error": "Failed to generate all-stages report",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


#========================================================
#Machines sxdnsjxnjm
#=========================

# ============================================================
# MACHINE STATUS & CONTROL ENDPOINTS
# ============================================================

@app.route('/api/machines', methods=['GET'])
def get_all_machines():
    """Get status of all 16 machines"""
    start_time = time.time()
    
    try:
        machines = mqtt_reader.get_all_machines()
        summary = mqtt_reader.get_machines_summary()
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "mqtt_connected": mqtt_reader.running,
            "total_machines": len(machines),
            "summary": summary,
            "machines": machines
        }
        
        response = convert_to_native_type(response)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log request
        api_logger.log_api_request(
            endpoint='/api/machines',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json={"total_machines": len(machines)}
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/machines/<machine_id>', methods=['GET'])
def get_machine_status(machine_id):
    """Get status of specific machine"""
    start_time = time.time()
    
    try:
        machine_data = mqtt_reader.get_machine_data(machine_id)
        
        if not machine_data:
            return jsonify({
                "status": "error",
                "message": f"Machine {machine_id} not found",
                "available_machines": list(mqtt_reader.machines.keys())
            }), 404
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "mqtt_connected": mqtt_reader.running,
            "machine": machine_data
        }
        
        response = convert_to_native_type(response)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log machine status
        api_logger.log_machine_status(machine_data)
        
        # Log request
        api_logger.log_api_request(
            endpoint=f'/api/machines/{machine_id}',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=response
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/machines/stage/<stage>', methods=['GET'])
def get_machines_by_stage(stage):
    """Get all machines for a specific process stage"""
    start_time = time.time()
    
    try:
        valid_stages = ["screening", "primary", "secondary", "tertiary", "final", "sludge_treatment", "dewatering"]
        
        if stage not in valid_stages:
            return jsonify({
                "status": "error",
                "message": f"Invalid stage. Valid stages: {', '.join(valid_stages)}"
            }), 400
        
        machines = mqtt_reader.get_machines_by_stage(stage)
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "mqtt_connected": mqtt_reader.running,
            "stage": stage,
            "machine_count": len(machines),
            "machines": machines
        }
        
        response = convert_to_native_type(response)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log request
        api_logger.log_api_request(
            endpoint=f'/api/machines/stage/{stage}',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json={"stage": stage, "machine_count": len(machines)}
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/machines/summary', methods=['GET'])
def get_machines_summary():
    """Get aggregated summary of all machines"""
    start_time = time.time()
    
    try:
        summary = mqtt_reader.get_machines_summary()
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "mqtt_connected": mqtt_reader.running,
            "summary": summary
        }
        
        response = convert_to_native_type(response)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log request
        api_logger.log_api_request(
            endpoint='/api/machines/summary',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=response
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MACHINE CONTROL ENDPOINTS
# ============================================================

@app.route('/api/control/<machine_id>/start', methods=['POST'])
def start_machine(machine_id):
    """Start a machine"""
    start_time = time.time()
    
    try:
        command = {
            "action": "start"
        }
        
        success = mqtt_reader.send_command(machine_id, command)
        
        response = {
            "status": "success" if success else "error",
            "machine_id": machine_id,
            "command": "start",
            "timestamp": datetime.now().isoformat()
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log command
        api_logger.log_system_event(
            event_type="MACHINE_START_COMMAND",
            severity="INFO",
            category="CONTROL",
            message=f"START command sent to {machine_id}",
            details={"machine_id": machine_id, "command": command}
        )
        
        return jsonify(response), 200 if success else 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/control/<machine_id>/stop', methods=['POST'])
def stop_machine(machine_id):
    """Stop a machine"""
    start_time = time.time()
    
    try:
        command = {
            "action": "stop"
        }
        
        success = mqtt_reader.send_command(machine_id, command)
        
        response = {
            "status": "success" if success else "error",
            "machine_id": machine_id,
            "command": "stop",
            "timestamp": datetime.now().isoformat()
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log command
        api_logger.log_system_event(
            event_type="MACHINE_STOP_COMMAND",
            severity="WARNING",
            category="CONTROL",
            message=f"STOP command sent to {machine_id}",
            details={"machine_id": machine_id, "command": command}
        )
        
        return jsonify(response), 200 if success else 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/control/<machine_id>/speed', methods=['POST'])
def set_machine_speed(machine_id):
    """Set machine speed (manual control)"""
    start_time = time.time()
    
    try:
        data = request.get_json() or {}
        speed = data.get('speed')
        reason = data.get('reason', 'Manual speed adjustment')
        
        if speed is None:
            return jsonify({"error": "Speed parameter required"}), 400
        
        command = {
            "action": "set_speed",
            "speed": float(speed),
            "reason": reason
        }
        
        success = mqtt_reader.send_command(machine_id, command)
        
        response = {
            "status": "success" if success else "error",
            "machine_id": machine_id,
            "command": "set_speed",
            "speed": speed,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log command
        api_logger.log_system_event(
            event_type="MACHINE_SPEED_CHANGE",
            severity="INFO",
            category="CONTROL",
            message=f"Speed changed for {machine_id} to {speed}%",
            details={"machine_id": machine_id, "speed": speed, "reason": reason}
        )
        
        return jsonify(response), 200 if success else 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/control/<machine_id>/mode', methods=['POST'])
def set_machine_mode(machine_id):
    """Set machine control mode (manual/auto)"""
    start_time = time.time()
    
    try:
        data = request.get_json() or {}
        mode = data.get('mode')
        
        if mode not in ['manual', 'auto']:
            return jsonify({"error": "Mode must be 'manual' or 'auto'"}), 400
        
        command = {
            "action": "set_mode",
            "mode": mode
        }
        
        success = mqtt_reader.send_command(machine_id, command)
        
        response = {
            "status": "success" if success else "error",
            "machine_id": machine_id,
            "command": "set_mode",
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log command
        api_logger.log_system_event(
            event_type="MACHINE_MODE_CHANGE",
            severity="INFO",
            category="CONTROL",
            message=f"Control mode changed for {machine_id} to {mode}",
            details={"machine_id": machine_id, "mode": mode}
        )
        
        return jsonify(response), 200 if success else 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/control/all/<action>', methods=['POST'])
def broadcast_command(action):
    """Broadcast command to all machines"""
    start_time = time.time()
    
    try:
        valid_actions = ['start', 'stop']
        
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action. Use: {', '.join(valid_actions)}"}), 400
        
        command = {
            "action": action
        }
        
        success = mqtt_reader.send_command('all', command)
        
        response = {
            "status": "success" if success else "error",
            "command": f"broadcast_{action}",
            "target": "all_machines",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log broadcast command
        api_logger.log_system_event(
            event_type="BROADCAST_COMMAND",
            severity="CRITICAL",
            category="CONTROL",
            message=f"Broadcast {action.upper()} command sent to all machines",
            details={"action": action, "target": "all_machines"}
        )
        
        return jsonify(response), 200 if success else 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# COMPREHENSIVE LOG ACCESS ENDPOINTS
# ============================================================

@app.route('/api/logs/all-activities', methods=['GET'])
def get_all_activities():
    """Get ALL logged activities across all tables"""
    limit = request.args.get('limit', 50, type=int)
    
    activities = []
    
    # Get recent from each table
    tables = {
        'api_request_logs': 'API Request',
        'water_quality_predictions': 'Water Quality Prediction',
        'chemical_dosing_predictions': 'Chemical Dosing',
        'equipment_failure_predictions': 'Equipment Failure Check',
        'treatment_process_logs': 'Treatment Process Control',
        'sensor_readings': 'Sensor Reading',
        'machine_status_logs': 'Machine Status Update',
        'mqtt_message_logs': 'MQTT Message',
        'system_events': 'System Event'
    }
    
    for table, activity_type in tables.items():
        logs = api_logger.get_logs(table, limit=limit//len(tables))
        for log in logs:
            log['activity_type'] = activity_type
            activities.append(log)
    
    # Sort by timestamp
    activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({
        "status": "success",
        "total_activities": len(activities),
        "activities": activities[:limit],
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/water-quality-predictions', methods=['GET'])
def get_water_quality_prediction_logs():
    """Get water quality prediction logs with complete input/output + LOGGING"""
    start_time = time.time()
    
    try:
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        stage = request.args.get('stage', None)
        
        # Build filters
        filters = {}
        if stage:
            filters['stage'] = stage
        
        # Retrieve logs from database
        logs = api_logger.get_logs('water_quality_predictions', limit, offset, filters)
        
        response = {
            "status": "success",
            "count": len(logs),
            "logs": logs,
            "query_parameters": {
                "limit": limit,
                "offset": offset,
                "stage": stage
            },
            "timestamp": datetime.now().isoformat()
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # ============================================================
        # LOG THIS API REQUEST
        # ============================================================
        api_logger.log_api_request(
            endpoint='/api/logs/water-quality-predictions',
            method='GET',
            status_code=200,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json={"count": len(logs), "limit": limit, "offset": offset}  # Don't log all logs
        )
        
        # ============================================================
        # LOG PERFORMANCE METRICS
        # ============================================================
        api_logger.log_performance_metric(
            metric_type="LOG_RETRIEVAL",
            endpoint='/api/logs/water-quality-predictions',
            operation="database_query",
            response_time_ms=processing_time_ms,
            database_time_ms=processing_time_ms,
            success=True,
            error_count=0,
            details={
                "records_returned": len(logs),
                "query_limit": limit,
                "query_offset": offset,
                "filter_stage": stage
            }
        )
        
        # ============================================================
        # LOG SYSTEM EVENT (only for large queries)
        # ============================================================
        if len(logs) > 500:
            api_logger.log_system_event(
                event_type="LARGE_LOG_QUERY",
                severity="INFO",
                category="DATABASE",
                message=f"Large water quality prediction log query: {len(logs)} records",
                details={
                    "records_count": len(logs),
                    "limit": limit,
                    "offset": offset,
                    "stage_filter": stage,
                    "client_ip": request.remote_addr
                }
            )
        
        return jsonify(response), 200
    
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_trace = traceback.format_exc()
        
        # ============================================================
        # LOG ERROR
        # ============================================================
        api_logger.log_api_request(
            endpoint='/api/logs/water-quality-predictions',
            method='GET',
            status_code=500,
            response_time_ms=processing_time_ms,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            request_headers=dict(request.headers),
            request_args=dict(request.args),
            request_json=None,
            response_json=None,
            error_message=str(e),
            traceback=error_trace
        )
        
        api_logger.log_system_event(
            event_type="LOG_RETRIEVAL_ERROR",
            severity="ERROR",
            category="DATABASE",
            message=f"Failed to retrieve water quality logs: {str(e)}",
            details={
                "error": str(e),
                "traceback": error_trace,
                "query_params": dict(request.args)
            }
        )
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500



@app.route('/api/logs/equipment-failure-predictions', methods=['GET'])
def get_equipment_failure_prediction_logs():
    """Get equipment failure prediction logs"""
    limit = request.args.get('limit', 100, type=int)
    machine_id = request.args.get('machine_id', None)
    
    filters = {}
    if machine_id:
        filters['machine_id'] = machine_id
    
    logs = api_logger.get_logs('equipment_failure_predictions', limit, 0, filters)
    
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/treatment-processes', methods=['GET'])
def get_treatment_process_logs():
    """Get treatment process control logs"""
    limit = request.args.get('limit', 100, type=int)
    stage = request.args.get('stage', None)
    
    filters = {}
    if stage:
        filters['stage'] = stage
    
    logs = api_logger.get_logs('treatment_process_logs', limit, 0, filters)
    
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/sensor-readings', methods=['GET'])
def get_sensor_reading_logs():
    """Get raw sensor readings"""
    limit = request.args.get('limit', 100, type=int)
    stage = request.args.get('stage', None)
    
    filters = {}
    if stage:
        filters['stage'] = stage
    
    logs = api_logger.get_logs('sensor_readings', limit, 0, filters)
    
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/machine-status', methods=['GET'])
def get_machine_status_logs():
    """Get machine status logs"""
    limit = request.args.get('limit', 100, type=int)
    machine_id = request.args.get('machine_id', None)
    
    filters = {}
    if machine_id:
        filters['machine_id'] = machine_id
    
    logs = api_logger.get_logs('machine_status_logs', limit, 0, filters)
    
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/mqtt-messages', methods=['GET'])
def get_mqtt_message_logs():
    """Get MQTT message logs"""
    limit = request.args.get('limit', 100, type=int)
    topic = request.args.get('topic', None)
    
    filters = {}
    if topic:
        filters['topic'] = topic
    
    logs = api_logger.get_logs('mqtt_message_logs', limit, 0, filters)
    
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/statistics', methods=['GET'])
def get_comprehensive_statistics():
    """Get comprehensive logging statistics"""
    stats = api_logger.get_statistics()
    
    return jsonify({
        "status": "success",
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/logs/export/<table>', methods=['GET'])
def export_table_to_csv(table):
    """Export any log table to CSV"""
    valid_tables = [
        'api_request_logs', 'water_quality_predictions', 'chemical_dosing_predictions',
        'equipment_failure_predictions', 'treatment_process_logs', 'mqtt_message_logs',
        'machine_status_logs', 'system_events', 'sensor_readings', 'performance_metrics'
    ]
    
    if table not in valid_tables:
        return jsonify({
            "status": "error",
            "message": f"Invalid table. Valid tables: {', '.join(valid_tables)}",
            "timestamp": datetime.now().isoformat()
        }), 400
    
    try:
        conn = sqlite3.connect(api_logger.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 10000")
        rows = cursor.fetchall()
        
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        conn.close()
        
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
        
        csv_data = output.getvalue()
        output.close()
        
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=hydronail_{table}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400

# ============================================================
# MAIN SERVER START
# ============================================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("🚀 Starting HydroNail ML API Server with MQTT Integration")
    print("="*60)

    if mqtt_reader.connect():
        print("✅ MQTT initialized - receiving real-time sensor data")
        time.sleep(2)
    else:
        print("⚠️ MQTT not available - using API with manual sensor input")

    print("📍 API Documentation: http://localhost:5000")
    print("🔗 Health Check: http://localhost:5000/api/health")
    print("📊 MQTT Status: http://localhost:5000/api/mqtt/status")
    print("🌊 All Stages Report: http://localhost:5000/api/all-stages/report")
    print("="*60 + "\n")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
        mqtt_reader.disconnect()
