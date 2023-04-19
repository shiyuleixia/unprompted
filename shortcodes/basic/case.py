class Shortcode():
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted
		self.description = "Use within [switch] to run different logic blocks depending on the value of a var."

	def run_block(self, pargs, kwargs, context, content):
		_var = self.Unprompted.shortcode_objects["switch"].switch_var

		# Default case
		if len(pargs) == 0 and _var != "":
			return(self.Unprompted.parse_alt_tags(content,context))
		# Matching case
		elif self.Unprompted.is_equal(_var,self.Unprompted.parse_advanced(pargs[0],context)):
			self.Unprompted.shortcode_objects["switch"].switch_var = ""
			return(self.Unprompted.parse_alt_tags(content,context))
		
		return("")

	def ui(self,gr):
		gr.Textbox(label="Matching value 🡢 str",max_lines=1)