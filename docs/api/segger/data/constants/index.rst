segger.data.constants
=====================

.. py:module:: segger.data.constants




Module Contents
---------------

.. py:class:: SpatialTranscriptomicsKeys

   Bases: :py:obj:`enum.Enum`


   Unified keys for spatial transcriptomics data, supporting multiple platforms.


   .. py:attribute:: TRANSCRIPTS_FILE


   .. py:attribute:: BOUNDARIES_FILE


   .. py:attribute:: CELL_METADATA_FILE


   .. py:attribute:: CELL_ID


   .. py:attribute:: TRANSCRIPTS_ID


   .. py:attribute:: TRANSCRIPTS_X


   .. py:attribute:: TRANSCRIPTS_Y


   .. py:attribute:: BOUNDARIES_VERTEX_X


   .. py:attribute:: BOUNDARIES_VERTEX_Y


   .. py:attribute:: GLOBAL_X


   .. py:attribute:: GLOBAL_Y


   .. py:attribute:: METADATA_CELL_KEY


   .. py:attribute:: COUNTS_CELL_KEY


   .. py:attribute:: CELL_X


   .. py:attribute:: CELL_Y


   .. py:attribute:: FEATURE_NAME


   .. py:attribute:: QUALITY_VALUE


   .. py:attribute:: OVERLAPS_BOUNDARY


.. py:class:: XeniumKeys

   Bases: :py:obj:`enum.Enum`


   Keys for *10X Genomics Xenium* formatted dataset.


   .. py:attribute:: TRANSCRIPTS_FILE
      :value: 'transcripts.parquet'



   .. py:attribute:: BOUNDARIES_FILE
      :value: 'nucleus_boundaries.parquet'



   .. py:attribute:: CELL_METADATA_FILE
      :value: None



   .. py:attribute:: CELL_ID
      :value: 'cell_id'



   .. py:attribute:: TRANSCRIPTS_ID
      :value: 'transcript_id'



   .. py:attribute:: TRANSCRIPTS_X
      :value: 'x_location'



   .. py:attribute:: TRANSCRIPTS_Y
      :value: 'y_location'



   .. py:attribute:: BOUNDARIES_VERTEX_X
      :value: 'vertex_x'



   .. py:attribute:: BOUNDARIES_VERTEX_Y
      :value: 'vertex_y'



   .. py:attribute:: FEATURE_NAME
      :value: 'feature_name'



   .. py:attribute:: QUALITY_VALUE
      :value: 'qv'



   .. py:attribute:: OVERLAPS_BOUNDARY
      :value: 'overlaps_nucleus'



   .. py:attribute:: METADATA_CELL_KEY
      :value: None



   .. py:attribute:: COUNTS_CELL_KEY
      :value: None



   .. py:attribute:: CELL_X
      :value: None



   .. py:attribute:: CELL_Y
      :value: None



.. py:class:: MerscopeKeys

   Bases: :py:obj:`enum.Enum`


   Keys for *MERSCOPE* data (Vizgen platform).


   .. py:attribute:: TRANSCRIPTS_FILE
      :value: 'detected_transcripts.csv'



   .. py:attribute:: BOUNDARIES_FILE
      :value: 'cell_boundaries.parquet'



   .. py:attribute:: CELL_METADATA_FILE
      :value: 'cell_metadata.csv'



   .. py:attribute:: CELL_ID
      :value: 'EntityID'



   .. py:attribute:: TRANSCRIPTS_ID
      :value: 'transcript_id'



   .. py:attribute:: TRANSCRIPTS_X
      :value: 'global_x'



   .. py:attribute:: TRANSCRIPTS_Y
      :value: 'global_y'



   .. py:attribute:: BOUNDARIES_VERTEX_X
      :value: 'center_x'



   .. py:attribute:: BOUNDARIES_VERTEX_Y
      :value: 'center_y'



   .. py:attribute:: FEATURE_NAME
      :value: 'gene'



   .. py:attribute:: QUALITY_VALUE
      :value: None



   .. py:attribute:: OVERLAPS_BOUNDARY
      :value: None



   .. py:attribute:: METADATA_CELL_KEY
      :value: 'EntityID'



   .. py:attribute:: COUNTS_CELL_KEY
      :value: 'cell'



   .. py:attribute:: CELL_X
      :value: 'center_x'



   .. py:attribute:: CELL_Y
      :value: 'center_y'



