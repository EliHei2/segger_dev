utils
=====

.. py:module:: utils






Module Contents
---------------

.. py:function:: add_options(config_path: os.PathLike, show_default: bool = True)

   A decorator to add command-line options to a Click command from a YAML
   configuration file.

   Parameters:
   config_path (os.PathLike): The path to the YAML configuration file
   containing the options.
   show_default (bool): Whether to show default values in help.

   Returns:
   function: A decorator function that adds the options to the Click command.

   The YAML configuration file should have the following format:
   ```
   option_name:
       type: "type_name"  # Optional, the type of the option
       (e.g., "str", "int")
       help: "description"  # Optional, the help text for the option
       default: value  # Optional, the default value for the option
       required: true/false  # Optional, whether the option is required
       ...
   ```
   Example usage:
   ```
   # config.yaml
   name:
       type: "str"
       help: "Your name"
       required: true
   age:
       type: "int"
       help: "Your age"
       default: 30

   # script.py
   @add_options('config.yaml')
   @click.command()
   def greet(args):
       click.echo(f"Hello, {args.name}! You are {args.age} years old.")
   ```


.. py:class:: CustomFormatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)

   Bases: :py:obj:`logging.Formatter`


   Custom logging formatter to add color-coded log levels to the log messages.

   Attributes:
   grey (str): ANSI escape code for grey color.
   yellow (str): ANSI escape code for yellow color.
   red (str): ANSI escape code for red color.
   bold_red (str): ANSI escape code for bold red color.
   reset (str): ANSI escape code to reset color.
   format (str): The format string for log messages.
   FORMATS (dict): A dictionary mapping log levels to their respective
   color-coded format strings.

   Methods:
   format(record):
       Format the specified record as text, applying color codes based on the
       log level.


   .. py:attribute:: grey
      :value: '\x1b[38;20m'



   .. py:attribute:: green
      :value: '\x1b[32;20m'



   .. py:attribute:: yellow
      :value: '\x1b[33;20m'



   .. py:attribute:: red
      :value: '\x1b[31;20m'



   .. py:attribute:: bold_red
      :value: '\x1b[31;1m'



   .. py:attribute:: reset
      :value: '\x1b[0m'



   .. py:attribute:: format
      :value: '%(asctime)s %(levelname)s: %(message)s'


      Format the specified record as text.

      The record's attribute dictionary is used as the operand to a
      string formatting operation which yields the returned string.
      Before formatting the dictionary, a couple of preparatory steps
      are carried out. The message attribute of the record is computed
      using LogRecord.getMessage(). If the formatting string uses the
      time (as determined by a call to usesTime(), formatTime() is
      called to format the event time. If there is exception information,
      it is formatted using formatException() and appended to the message.


   .. py:attribute:: FORMATS


