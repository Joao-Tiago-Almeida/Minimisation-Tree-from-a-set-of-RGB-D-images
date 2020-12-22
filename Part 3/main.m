desired_line = 1;

fid = fopen('output.txt');
tline = fgetl(fid);
current_line = 0;
while ischar(tline)
    tline = fgetl(fid);
    if(current_line == desired_line)
      break
    end
    current_line += 1;
end
%disp(tline)

for val = split(" ")
  val 
  "\n"  
end

fclose(fid);