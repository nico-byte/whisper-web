Whisper Web API Reference
=========================

This section provides a comprehensive reference for all modules and classes in the Whisper Web real-time transcription system.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core Components
===============

Server & API
-------------

.. automodule:: whisper_web.server
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:

The main FastAPI server that provides RESTful endpoints and WebSocket connections for real-time transcription services. Handles session management, audio streaming, and transcription delivery.

Transcription and Audio Management
------------------------

.. automodule:: whisper_web.management
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:

Event-driven transcription manager that coordinates the entire transcription pipeline. Manages audio queues, processes chunks, and delivers completed transcriptions through the event system.

Speech Recognition
==================

Whisper Model
-----------------------

.. automodule:: whisper_web.whisper_model
   :members:
   :undoc-members:
   :show-inheritance:

Wrapper for OpenAI's Whisper models with configuration management and device optimization. Provides the core speech-to-text functionality with support for various model sizes and configurations.

Audio Processing
================

Audio Input Stream Generator
----------------------------

.. automodule:: whisper_web.inputstream_generator
   :members:
   :undoc-members:
   :show-inheritance:

Handles real-time audio capture and processing. Converts audio streams into chunks suitable for transcription, with configurable sample rates, chunk sizes, and audio preprocessing options.

Event System
============

Events & Event Bus
------------------

.. automodule:: whisper_web.events
   :members:
   :undoc-members:
   :show-inheritance:

Asynchronous event system that coordinates communication between components. Defines all event types used throughout the transcription pipeline for loose coupling and extensibility.

Data Types & Utilities
======================

Core Data Types
---------------

.. automodule:: whisper_web.types
   :members:
   :undoc-members:
   :show-inheritance:

Fundamental data structures used throughout the system, including audio chunks and transcription objects with proper typing and validation.

Utility Functions
-----------------

.. automodule:: whisper_web.utils
   :members:
   :undoc-members:
   :show-inheritance:

Helper functions for device management, configuration, and other common operations used across the transcription system.