module Model
using Distributions: wsample
using DataStructures: DefaultDict

# Necessary Changes
# A visit must last at least 2 hours
# Multiple visits on the same day still count as a single visit
# Look at what special cases Moe removes
# Look at if weber optimality holds for those that
# come from farther than x away (I think- review paper)

# Need to find:
# weber optimality
  # for every cell, keep home location -> frequency
# freq_dist_count (frequency, distance -> avg #people)
# energy per cell

# Explore a new location, distance sampled by truncated Pareto
const alpha = 0.55
const h = 100
const top = h^alpha
@inline function sampleDisplacement()
  u = rand()
  return (-((u * (top - 1) - top) / top))^(-1/alpha)
end

# Wait a given amount of time sampled by a truncated Pareto
const beta = 0.8
const h2 = 17
const top2 = h2^beta
@inline function sampleDelay(n)
  u = rand(n)
  return (-((u .* (top2 - 1) - top2) / top2)).^(-1/beta)
end

# Pick a random direction and jump r sampled pareto
@inline function randomJump(l)
  r = sampleDisplacement()
  theta = rand() * 2 * pi
  h = [cos(theta), sin(theta)]
  l .+ (h .* sampleDisplacement())
end

# group by keys and sum
function monoidDict(keys, vals)
  d = DefaultDict(typeof(keys[1]), typeof(vals[1]), 0)
  for i in 1:size(keys)[1]
    d[keys[i]] += vals[i]
  end
  d
end

const rho = 0.6
const gamma = 0.21
const monthHours = 24 * 30

function walk(n)
  energy = zeros(100, 100)
  guestbook = [DefaultDict{Vector{Float32},Int}(0) for i in 1:100, j in 1:100]
  nextjump = zeros(Float32,n)
  homes = [Float32.(rand(0:300,2)) for _ in 1:n]
  agentLocs = copy(homes)
  visits = [[1] for _ in 1:n]
  visitLocs = [[homes[i]] for i in 1:n]
  time = 0.0
  while time < monthHours
    nextjump[:] .= time .+ sampleDelay(n)
    perm = sortperm(nextjump)
    for i in perm
      time = nextjump[i]
      l = agentLocs[i]
      gridL = floor.(Int, l)
      if all(gridL .> 100) && all(gridL .<= 200)
        en = norm(homes[i] - l)
        gridCell = gridL .- [100,100]
        energy[gridCell...] += en
        guestbook[gridCell...][homes[i]] += 1
      end
      if rand() < rho * length(visits[i]) ^ -gamma
        push!(visits[i], 1)
        agentLocs[i] = randomJump(l)
        push!(visitLocs[i], agentLocs[i])
      else
        locid = wsample(visits[i].^2)
        visits[i][locid] += 1
        agentLocs[i] = visitLocs[i][locid]
      end
    end
  end
  energy, guestbook, homes, visits, visitLocs
end

function fdc(visitLocs, visits, homes)
  freq_dist_count = zeros(Int, 10, 50)
  for i in 1:length(visits)
    for (k,v) in zip(visitLocs[i], visits[i])
      if all(k .> 100) && all(k .<= 200)
        halfdist = trunc(Int, norm(homes[i] - k) / 2)
        if halfdist < 50 && v > 0 && v <= 10
          freq_dist_count[v, halfdist + 1] += 1
        end
      end
    end
  end
  freq_dist_count
end

centroid(d) =
  sum(k * v for (k, v) in d) / sum(v for (_,v) in d)

cellEnergySq(d, l) = sum(norm(l - k)^2 * v for (k, v) in d)
cellEnergy(d, l) = sum(norm(l - k) * v for (k, v) in d)

function energyGapSq(homeLoc, dict)
  current = cellEnergySq(dict, homeLoc)
  optimal = cellEnergySq(dict, centroid(dict))
  (current - optimal) / current
end

function energyGap(homeLoc, dict)
  current = cellEnergy(dict, homeLoc)
  optimal = weberOpt(dict, homeLoc, current)
  (current - optimal) / current
end

function weberOpt(dict, homeLoc, current)
  dirs = [0 1; 1 0; 0 -1; -1 0]'
  while true
    x, nth = findmin(cellEnergy(dict, homeLoc + dirs[:,ix])
      for ix in 1:4)
    if x < current
      current = x
      homeLoc .+= @view dirs[:,nth]
    else
      break
    end
  end
  current
end

function optGapsSq(guestbook) 
  [energyGapSq(collect(k.I) + [100,100],v) for (k,v) in
    enumerate(IndexCartesian(), guestbook)]
end

function optGaps(guestbook) 
  [energyGap(collect(k.I) + [100,100],v) for (k,v) in
    enumerate(IndexCartesian(), guestbook)]
end

function main()
  energy, guestbook, homes, visits, visitLocs = walk(1000)
  plot(x=optGaps(guestbook), Geom.histogram)
  freqdist = fdc(visitLocs, visits)
  for f in 1:5
    xaxis = (1:50) * 2 * f^2
    mask = (@view freqdist[f,:]) .> 0
    plot(x=xaxis[mask], y=freqdist[f,mask], Geom.point,
      Scale.x_log10, Scale.y_log10)
  end
end

end
