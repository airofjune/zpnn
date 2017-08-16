local THNN = require 'nn.THNN'
local LSTM, parent = torch.class('nn.LSTM', 'nn.Module')

--[[
input:  X(t), H(t-1), C(t-1)
output: H(t), C(t)
F(t) = sig{Whf*H(t-1) + Wxf*X(t)}
I(t) = sig{Whi*H(t-1) + Wxi*X(t)}
O(t) = sig{Who*H(t-1) + Wxo*X(t)}
C_(t) = sig{Whc*H(t-1) + Wxc*X(t)}
C(t) = F(t) .* C(t-1) + I(t) .* C_(t)
H(t) = O(t) .* tan{C(t)}
--]]

function LSTM:__init(inputSize, hiddenSize, bias)
   parent.__init(self)
   outputSize = 4 * hiddenSize
   self.hs = hiddenSize
   local bias = ((bias == nil) and true) or bias
   self.output_c = torch.Tensor()
   self.output_h = torch.Tensor()
   self.output = {self.output_c, self.output_h}
   self.grad_in_c = torch.Tensor()
   self.grad_in_h = torch.Tensor()
   self.grad_in_x = torch.Tensor()
   self.gradInput = {self.grad_in_c, self.grad_in_h, self.grad_in_x}
   self.weight     = torch.Tensor(hiddenSize + inputSize, outputSize)
   self.gradWeight = torch.Tensor(hiddenSize + inputSize, outputSize)
   if bias then
      self.bias     = torch.Tensor(2*outputSize)
      self.gradBias = torch.Tensor(2*outputSize)
   end
   self.dnnPrimitives = torch.Tensor(30)
   self.inMem = torch.Tensor()
   self:reset()
end

--to compare, initalize weight for h and x
function LSTM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

function LSTM:updateOutput(input)
   batch_size = input[1]:size(1)
   self.output_c:resize(batch_size, self.hs)
   self.output_h:resize(batch_size, self.hs)
   --allocate internal memory for LSTM.c using bs*(3*hs4+hs+hs) + hs4
   self.inMem:resize(batch_size*self.hs*14+self.hs*4)
   input[1].THNN.LSTM_updateOutput(
      self.dnnPrimitives:cdata(),
      self.inMem:cdata(),
      input[1]:cdata(),
      input[2]:cdata(),
      input[3]:cdata(),
      self.output_c:cdata(),
      self.output_h:cdata(),
      self.weight:cdata(),
      self.bias:cdata()
   )
   return self.output
end

function LSTM:updateGradInput(input, gradOutput)
   batch_size = input[1]:size(1)
   self.grad_in_c:resize(batch_size, input[1]:size(2))
   self.grad_in_h:resize(batch_size, input[2]:size(2))
   self.grad_in_x:resize(batch_size, input[3]:size(2))

   input[1].THNN.LSTM_updateGradInput(
      self.dnnPrimitives:cdata(),
      input[1]:cdata(),
      input[2]:cdata(),
      input[3]:cdata(),
      self.weight:cdata(),
      gradOutput[1]:cdata(),  --c
      gradOutput[2]:cdata(),  --h
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.grad_in_c:cdata(),
      self.grad_in_h:cdata(),
      self.grad_in_x:cdata()
   )
   return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
end

function LSTM:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function LSTM:clearState()
   return parent.clearState(self)
end

function LSTM:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
