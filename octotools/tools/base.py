# octotools/tools/base.py

from octotools.engine.openai import ChatOpenAI

class BaseTool:
    """
    A base class for building tool classes that perform specific tasks, such as image processing or text detection.
    Supports both API and local LLM models.
    """

    require_llm_engine = False  # Default is False, tools that need LLM should set this to True

    def __init__(self, tool_name=None, tool_description=None, tool_version=None, 
                 input_types=None, output_type=None, demo_commands=None, 
                 output_dir=None, user_metadata=None, model_string=None,
                 use_local_model=False, capture_logits=False, logits_dir=None):
        """
        Initialize the base tool with optional metadata.

        Parameters:
            tool_name (str): The name of the tool.
            tool_description (str): A description of the tool.
            tool_version (str): The version of the tool.
            input_types (dict): The expected input types for the tool.
            output_type (str): The expected output type for the tool.
            demo_commands (list): A list of example commands for using the tool.
            output_dir (str): The directory where the tool should save its output (optional).
            user_metadata (dict): Additional metadata specific to user needs (optional).
            model_string (str): The model string for the LLM engine (optional, only used if require_llm_engine is True).
            use_local_model (bool): Whether to use a local model instead of API (default: False).
            capture_logits (bool): Whether to capture logits for local models (default: False).
            logits_dir (str): Directory to store captured logits for local models (optional).
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands
        self.output_dir = output_dir
        self.user_metadata = user_metadata
        
        # LLM-related attributes
        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir

    def set_metadata(self, tool_name, tool_description, tool_version, input_types, output_type, demo_commands, user_metadata=None):
        """
        Set the metadata for the tool.

        Parameters:
            tool_name (str): The name of the tool.
            tool_description (str): A description of the tool.
            tool_version (str): The version of the tool.
            input_types (dict): The expected input types for the tool.
            output_type (str): The expected output type for the tool.
            demo_commands (list): A list of example commands for using the tool.
            user_metadata (dict): Additional metadata specific to user needs (optional).
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands
        self.user_metadata = user_metadata

    def get_metadata(self):
        """
        Returns the metadata for the tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = {
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "tool_version": self.tool_version,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "demo_commands": self.demo_commands,
            "require_llm_engine": self.require_llm_engine,
        }
        
        # Add LLM-specific metadata if the tool requires LLM
        if self.require_llm_engine:
            metadata["llm_config"] = {
                "model_string": self.model_string,
                "model_type": "local" if self.use_local_model else "API",
                "capture_logits": self.capture_logits if self.use_local_model else None,
                "logits_dir": self.logits_dir if self.use_local_model and self.capture_logits else None
            }
            
        if self.user_metadata:
            metadata["user_metadata"] = self.user_metadata
        return metadata

    def set_custom_output_dir(self, output_dir):
        """
        Set a custom output directory for the tool.

        Parameters:
            output_dir (str): The new output directory path.
        """
        self.output_dir = output_dir

    def set_llm_config(self, model_string=None, use_local_model=False, 
                      capture_logits=False, logits_dir=None):
        """
        Set the LLM configuration for the tool.

        Parameters:
            model_string (str): The model string for the LLM engine.
            use_local_model (bool): Whether to use a local model instead of API.
            capture_logits (bool): Whether to capture logits for local models.
            logits_dir (str): Directory to store captured logits for local models.
        """
        self.model_string = model_string
        self.use_local_model = use_local_model
        self.capture_logits = capture_logits
        self.logits_dir = logits_dir

    def create_llm_engine(self, is_multimodal=False):
        """
        Create and return a new LLM engine instance with the current configuration.

        Parameters:
            is_multimodal (bool): Whether the LLM engine should support multimodal inputs.

        Returns:
            ChatOpenAI: A configured LLM engine instance.
        """
        if not self.require_llm_engine:
            raise ValueError("This tool does not require an LLM engine.")
            
        return ChatOpenAI(
            model_string=self.model_string,
            is_multimodal=is_multimodal,
            use_local_model=self.use_local_model,
            capture_logits=self.capture_logits,
            logits_dir=self.logits_dir
        )

    def execute(self, *args, **kwargs):
        """
        Execute the tool's main functionality. This method should be overridden by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")