module Model
using Distributions: wsample
import DataStructures
using Images
using PyPlot

# Necessary Changes
# A visit must last at least 2 hours
# Multiple visits on the same day still count as a single visit
# Look at what special cases Moe removes

# Explore a new location, distance sampled by truncated Pareto
const alpha = 0.55
const h = 100
const top = h^alpha
function sampleDisplacement()
  u = rand()
  return (-((u * (top - 1) - top) / top))^(-1/alpha)
end

# Wait a given amount of time sampled by a truncated Pareto
const beta = 0.8
const h2 = 17
const top2 = h2^beta
function sampleDelay(n)
  u = rand(n)
  return (-((u .* (top2 - 1) - top2) / top2)).^(-1/beta)
end

# Get all grid cells within radius r
function gridCells(loc, rr) :: Array{Array{Int,1}, 1}
  cells = Array{Array{Int,1}, 1}()
  angle = 1.0 / r
  for theta in angle:angle:(2*pi - angle)
    rot = [cos(theta), sin(theta)]
    push!(cells, trunc.(Int, loc + rot * r))
  end
  cells
end

# Sample grid cells within radius r weighted by allVisits
function sampleJump(loc, allVisits, r)
  whichCells = gridCells(loc, r)
  unnormalized = getindex.(allVisits, whichCells) + 1
  sqvals = unnormalized .^ 2
  cell = wsample(sqvals)
  whichCells[cell]
end

# group by keys and sum
function monoidDict(keys, vals)
    d = DataStructures.DefaultDict(typeof(keys[1]), typeof(vals[1]), 0)
    for i in 1:size(keys)[1]
        d[keys[i]] += vals[i]
    end
    d
end

# Pick a random direction and jump r sampled pareto
function randomJump(l, r)
    theta = rand() * 2 * pi
    h = [cos(theta), sin(theta)]
    l + h * r
end

# Remember: how did the diffused stuff work?

function sampleDiff(diffused, r, l)
    w = round(Int, r)
    s = 2*w+1
    options = reshape(diffused[l[1]-w:l[1]+w, l[2]-w:l[2]+w], s^2) + 1
    o  = rand(Distributions.Categorical(options / sum(options))) - 1
    l + ([o % s, div(o, s)] - [w,w])
end

const rho = 0.6
const gamma = 0.21
const monthHours = 24 * 30

function walk(n)
    energy = zeros(100, 100)
    diffused = zeros(300, 300)
    nextjump = zeros(Float32,n)
    seen = ones(Int,n)
    homes = rand(140:160,n,2)
    locs = copy(homes)
    visits = [[1] for _ in 1:n]
    visitLoc = [[locs[i,:]] for i in 1:n]
    time =0.0
    while time .< monthHours
        nextjump = time + sampleDelay(n)
        perm = sortperm(nextjump)
        for j in 1:n
            i = perm[j]
            time = nextjump[i]
            l = locs[i,:]
            en = norm(homes[i,:] - l)
            diffused[l[1]-2:l[1]+2,l[2]-2:l[2]+2] += en
            diffused[l[1]-1:l[1]+1,l[2]-1:l[2]+1] += en
            diffused[l...] += en
            if all(l .>= 100) && all(l .< 200)
                energy[(l - 99)...] += en
            end
            if rand() < rho * seen[i] ^ -gamma
                seen[i] += 1
                push!(visits[i], 1)
                r = sampleDisplacement()
                loc = sampleDiff(diffused, r, l)
                push!(visitLoc[i], loc)
                locs[i,:] = copy(loc)
            else
                sqvisits = visits[i].^2
                locid = rand(Distributions.Categorical(sqvisits / sum(sqvisits)))
                visits[i][locid] += 1
                locs[i,:] = copy(visitLoc[i][locid])
            end
        end
    end
    freq_dist_count = zeros(Int, 10, 50)
    for i in 1:n
        for (k,v) in monoidDict(visitLoc[i], visits[i])
            if all(k .>= 100) && all(k .< 200)
                halfdist = trunc(Int, norm(homes[i,:] - k) / 2)
                if halfdist < 50 && v > 0 && v <= 10
                    freq_dist_count[v, halfdist + 1] += 1
                end
            end
        end
    end
    energy, homes, freq_dist_count
end

end
