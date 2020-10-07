my_phonebook = Dict("Jenny"=>"867-5309", "Ghostbusters" => "555-2368")

my_phonebook["Kramer"] = "555-FILK"

my_phonebook

my_phonebook["Jenny"]

pop!(my_phonebook,"Kramer")

my_phonebook

my_favorite_animals = ("penguins", "cats", "sugargliders")

my_favorite_animals[1]

my_friends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]

fibonacci = [1, 1, 2, 3, 5, 8, 13]

my_friends[3] = "baby bop"

my_friends

push!(fibonacci, 21)

pop!(fibonacci)

numbers = [[1,2,3],[4,5],[6,7,8,9]]

rand(4,3)

rand(4, 3, 2)

n = 0
while n < 10
    global n += 1
    println(n)
end

function plus_equals()
    n = 0
    while n < 10
        n += 1
        println(n)
    end
end

plus_equals()

i = 1
while i <= length(my_friends)
    friend = my_friends[i]
    println("Hi $friend, it's great to see you!")
    global i += 1
end

for n in 1:10
    println(n)
end

for friend in my_friends
    println("Hi $friend, it's great to see you!")
end

m, n = 5, 5
A = zeros(m,n)

for i in 1:m
    for j in 1:n
        A[i, j] = i + j
    end
end

A

B = zeros(m, n)
for i in 1:m, j in 1:n
    B[i, j] = i + j
end

B
A == B

C = [i + j for i in 1:m, j in 1:n]
C == A

for n in 1:10
    A = [i + j for i in 1:n, j in 1:n]
    display(A)
end

x = 3
y = 17
if x > y
    println("$x is larger than $y")
elseif y > x
    println("$y is larger than $x")
else
    println("$x and $y are equal!")
end

function sayhi(name)
    println("Hi $name, it's great to see you!")
end

sayhi("C-3PO")

function f(x)
    x^2
end

f(42)

sayhi2(name) = println("Hi $name, it's great to see you!")

sayhi2("Jimmy")

f2(x) = x^2

f2(41)

sayhi3 = name -> println("Go home, $name.")

sayhi3("Timmy")

f3 = x->x^2

f3(19)

A = rand(3,3)

f(A)

v = [3, 5, 2]

sort(v)

v

sort!(v)

v

A = [i + 3*j for j in 0:2, i in 1:3]

f(A)
f.(A)
f.(v)

using Example

hello("it's me. I was wondering if after all these years you'd like to meet.")

using Colors

palette = distinguishable_colors(100)

rand(palette, 3, 3)

using Plots

x = -3:0.1:3

f(x) = x^2

y = f.(x)

gr()

plot(x, y, label="line")
scatter!(x, y, label="points")

plotlyjs()

plot(x, y, label="line")
scatter!(x, y, label="points")

globaltemperatures = [14.14, 14.5, 14.8, 15.2, 15.5, 15.8]
numpirates =[45000, 20000, 15000, 5000, 400, 17]

plot(numpirates, globaltemperatures, legend = false)
scatter!(numpirates, globaltemperatures, legend = false)

xflip!()
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")

plot1 = plot(x, x)
plot2 = plot(x, x.^2)
plot3 = plot(x, x.^3)
plot4 = plot(x, x.^4)
plot(plot1, plot2, plot3, plot4, layout=(2,2) ,legend = false)

methods(+)

@which 3 + 3

@which 3.0 + 3.0

@which 3 + 3.0

import Base: +

+(x::String, y::String) = string(x,y)

"hello " + "world"
@which "hello " + "world"

foo(x,y) = println("duck-typed foo!")

foo(x::Int, y::Float64) = println("foo with an integer and a float!")

foo(x::Float64, y::Float64) = println("foo with two floats!")

foo(x::Int, y::Int) = println("foo with two integers!")

foo(1, 1)

foo(1., 1.)

foo(1, 1.0)

foo(2, "two")

a = rand(10^7)

sum(a)

using BenchmarkTools

A = rand(1:4, 3, 3)

B = A
C = copy(A)

[B C]

A[1] = 17

[B C]

x = ones(3)

b=A*x

A'

Asym = A + A'

Apd = A'A

A\b

Atall = A[:,1:2]
display(Atall)

Atall\b
